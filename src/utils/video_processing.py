import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import threading
import queue
import time
import math
from datetime import datetime
import os

# Initialize MediaPipe components
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def resize_preserve_aspect_ratio(image, target_size):
    """
    Resize image while preserving aspect ratio, then pad to target size.

    Args:
        image: Input image (numpy array)
        target_size: Desired output size as (width, height)

    Returns:
        Resized and padded image
    """
    target_width, target_height = target_size
    height, width = image.shape[:2]

    # Calculate the ratio of the target dimensions to the original dimensions
    width_ratio = target_width / width
    height_ratio = target_height / height

    # Use the smaller ratio to ensure the image fits within the target dimensions
    ratio = min(width_ratio, height_ratio)

    # Calculate new dimensions
    new_width = int(width * ratio)
    new_height = int(height * ratio)

    # Resize the image
    resized = cv2.resize(image, (new_width, new_height))

    # Create a black canvas of the target size
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # Calculate offsets to center the image
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    # Place the resized image on the canvas
    canvas[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = resized

    return canvas


def get_bounding_box(landmarks, width, height, margin=20):
    """
    Compute bounding box from hand landmarks with adjustable margin.

    Args:
        landmarks: MediaPipe hand landmarks object.
        width: Frame width.
        height: Frame height.
        margin: Pixel margin to add around the detected hand.

    Returns:
        tuple: (x_min, y_min, x_max, y_max) coordinates.
    """
    x_coords = [lm.x * width for lm in landmarks.landmark]
    y_coords = [lm.y * height for lm in landmarks.landmark]

    x_min = max(0, int(min(x_coords)) - margin)
    y_min = max(0, int(min(y_coords)) - margin)
    x_max = min(width, int(max(x_coords)) + margin)
    y_max = min(height, int(max(y_coords)) + margin)

    return x_min, y_min, x_max, y_max


class VideoProcessor:
    """Video processing class for handling webcam input and hand tracking."""

    def __init__(
        self,
        camera_id=0,
        width=640,
        height=480,
        max_hands=2,
        crop_size=(224, 224),
        resize_method="fixed",
    ):
        """
        Initialize the video processor.

        Args:
            camera_id: Camera device ID (default: 0)
            width: Frame width
            height: Frame height
            max_hands: Maximum number of hands to detect
            crop_size: Size to crop hand images to
            resize_method: Method for resizing ("fixed" or "preserve_ratio")
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap = None
        self.stop_event = threading.Event()
        self.thread = None
        self.current_frame = None

        # Configuration
        self.MAX_HANDS = max_hands
        self.CROP_SIZE = crop_size
        self.WRIST_IDX = 0
        self.RESIZE_METHOD = resize_method

        # MediaPipe hands setup
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.MAX_HANDS,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Record state
        self.is_recording = False
        self.recorded_frames = []
        self.skeletal_data = []
        self.crops_data = []
        self.no_hands_counter = 0
        self.frame_counter = 0
        self.last_hands_present = False
        self.current_frame_has_hands = False
        self.recording_just_ended = False
        self.auto_save_pending = False

    def start(self):
        """Start the video capture thread."""
        if self.thread is not None and self.thread.is_alive():
            return True  # Already running

        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            return False

        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Clear stop event
        self.stop_event.clear()

        # Start capture thread
        self.thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        """Stop the video capture thread."""
        if self.thread is None:
            return

        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def start_recording(self, no_hands_threshold=30):
        """
        Start recording frames and hand data.

        Args:
            no_hands_threshold: Frames without hands before stopping
        """
        self.is_recording = True
        self.recorded_frames = []
        self.skeletal_data = []
        self.crops_data = []
        self.frame_counter = 0
        self.no_hands_counter = 0
        self.no_hands_threshold = no_hands_threshold
        self.last_hands_present = False

    def stop_recording(self):
        """Stop recording and return the data."""
        self.is_recording = False
        return {
            "frames": len(self.recorded_frames),
            "skeletal_data": np.array(self.skeletal_data)
            if self.skeletal_data
            else np.array([]),
            "crops": self.crops_data,
        }

    def get_latest_frame(self):
        """Get the latest processed frame."""
        return self.current_frame

    def auto_record_if_needed(self, glosses_available, no_hands_threshold=30):
        """
        Automatically start recording if conditions are met (hands visible and glosses available).

        Args:
            glosses_available: Whether glosses have been entered by the user
            no_hands_threshold: Frames without hands before stopping

        Returns:
            dict: Status information about recording state changes
        """
        # Get current state
        has_hands = (
            self.current_frame_has_hands
            if hasattr(self, "current_frame_has_hands")
            else False
        )
        status = {"started": False, "stopped": False, "saved": False, "reason": None}

        # If glosses are available and hands are detected, start recording if not already recording
        if glosses_available and has_hands and not self.is_recording:
            self.start_recording(no_hands_threshold=no_hands_threshold)
            status["started"] = True
            status["reason"] = "Hands detected and glosses provided"

        # If recording has stopped automatically, report it
        if hasattr(self, "recording_just_ended") and self.recording_just_ended:
            status["stopped"] = True
            status["reason"] = "No hands detected"
            self.recording_just_ended = False

        return status

    def _capture_frames(self):
        """Thread function for continuous frame capture and processing."""
        while not self.stop_event.is_set():
            if self.cap is None or not self.cap.isOpened():
                break

            ret, frame = self.cap.read()
            if not ret:
                break

            # Flip frame for selfie view
            frame = cv2.flip(frame, 1)

            # Process with MediaPipe
            processed_frame, has_hands, hand_data = self._process_frame(frame)

            # Handle recording state
            if self.is_recording:
                # Check if hands are present
                if has_hands:
                    # Reset counter if hands reappear
                    self.no_hands_counter = 0

                    # Store frame and data
                    self.recorded_frames.append(frame.copy())

                    # Store skeletal data and crops
                    if hand_data:
                        self.skeletal_data.append(hand_data["skeletal"])
                        self.crops_data.append(hand_data["crops"])

                    # Increment frame counter
                    self.frame_counter += 1

                    # Update last state
                    self.last_hands_present = True
                elif self.last_hands_present:  # Hands were present but now gone
                    self.no_hands_counter += 1
                    if self.no_hands_counter >= self.no_hands_threshold:
                        # Automatically stop recording
                        self.is_recording = False
                        self.recording_just_ended = True
                        self.auto_save_pending = True  # Mark for auto-save
                else:
                    # No hands detected yet, wait for hands to appear
                    pass

            # Update current frame
            self.current_frame = processed_frame

            # Small delay to reduce CPU usage
            time.sleep(0.01)

    def _process_frame(self, frame):
        """
        Process a frame with MediaPipe Hands.

        Args:
            frame: Input frame

        Returns:
            Tuple of (processed_frame, has_hands, hand_data)
        """
        # Create a copy for annotations
        annotated_frame = frame.copy()
        h, w, _ = frame.shape

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        # Process with MediaPipe
        results = self.hands.process(rgb_frame)

        # Make image writeable again
        rgb_frame.flags.writeable = True

        # Check if hands are present and store state
        has_hands = results.multi_hand_landmarks and results.multi_handedness
        self.current_frame_has_hands = has_hands

        # Data storage
        hand_data = None

        if has_hands:
            # Initialize data structures
            frame_skeletal = np.zeros((self.MAX_HANDS, 21, 3))
            frame_crops = np.zeros((self.MAX_HANDS, *self.CROP_SIZE, 3), dtype=np.uint8)

            # Process each detected hand
            for hand_idx, (hand_landmarks, handedness) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_handedness)
            ):
                if hand_idx >= self.MAX_HANDS:
                    break  # Skip if exceeding max hands

                # Get hand type and confidence
                hand_type = handedness.classification[0].label
                hand_confidence = handedness.classification[0].score

                # Set color based on hand type
                color = (0, 0, 255) if hand_type == "Left" else (0, 255, 0)

                # Draw landmarks on the image
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=color, thickness=2),
                )

                # Extract landmarks
                skeletal = {}
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    skeletal[idx] = [landmark.x, landmark.y, landmark.z]

                # Step 1: Normalize relative to wrist (translation)
                wrist = skeletal[0]  # WRIST_IDX = 0 in MediaPipe
                for idx in skeletal:
                    skeletal[idx] = [skeletal[idx][i] - wrist[i] for i in range(3)]

                # Step 2: Scale normalization using wrist-to-middle-finger-MCP distance
                ref_vec = [
                    skeletal[9][i] - skeletal[0][i] for i in range(3)
                ]  # Wrist (0) to middle MCP (9)
                ref_length = math.sqrt(sum(v * v for v in ref_vec))  # Euclidean norm
                if ref_length > 0:
                    for idx in skeletal:
                        skeletal[idx] = [
                            skeletal[idx][i] / ref_length for i in range(3)
                        ]

                # (Optional) Step 3: Depth-specific normalization
                z_max = max(abs(skeletal[idx][2]) for idx in skeletal)
                if z_max > 0:
                    for idx in skeletal:
                        skeletal[idx][2] /= z_max

                # Store skeletal data
                for idx in skeletal:
                    frame_skeletal[hand_idx, idx] = skeletal[idx]

                # Get bounding box
                x_min, y_min, x_max, y_max = get_bounding_box(hand_landmarks, w, h, 20)

                # Extract and resize crop
                if x_max > x_min and y_max > y_min:  # Valid bounding box
                    crop = frame[y_min:y_max, x_min:x_max].copy()
                    if crop.size > 0:
                        if self.RESIZE_METHOD == "preserve_ratio":
                            crop = resize_preserve_aspect_ratio(crop, self.CROP_SIZE)
                        else:
                            crop = cv2.resize(crop, self.CROP_SIZE)
                        frame_crops[hand_idx] = crop

                # Draw rectangle and label
                cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(
                    annotated_frame,
                    f"{hand_type} ({hand_confidence:.2f})",
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            # Store hand data
            hand_data = {"skeletal": frame_skeletal, "crops": frame_crops}

        # Add recording indicator if recording
        if self.is_recording:
            cv2.putText(
                annotated_frame,
                f"Recording... Frame: {self.frame_counter}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

            if not has_hands and self.last_hands_present:
                cv2.putText(
                    annotated_frame,
                    f"No hands! Recording will end in: {self.no_hands_threshold - self.no_hands_counter}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
        elif not has_hands:
            cv2.putText(
                annotated_frame,
                "Waiting for hands to appear...",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        return annotated_frame, has_hands, hand_data
