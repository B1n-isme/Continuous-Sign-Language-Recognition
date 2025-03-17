import sys
import cv2
import numpy as np
import math
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QHBoxLayout,
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
import mediapipe as mp
from input_modalities.optical_flow_raft import compute_optical_flow
import os
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
mp_drawing = mp.solutions.drawing_utils


def to_base64(num):
    """
    Convert a number to base64 encoding using only filename-safe characters.

    Args:
        num (int): The number to convert

    Returns:
        str: Base64 encoded string safe for filenames
    """
    # Use a filename-safe character set (no '/' or '+' that might be in standard base64)
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
    if num == 0:
        return "0"

    result = ""
    while num > 0:
        result = chars[num % 64] + result
        num //= 64

    return result


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
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a black canvas of the target size
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # Calculate offsets to center the image
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    # Place the resized image on the canvas
    canvas[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = resized

    return canvas


class SignLanguageCapture(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Sign Language Capture")

        # Initialize camera with higher resolution
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # Get camera's native resolution
        self.cam_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cam_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate a reasonable video display size that isn't too dominant
        # Use 70% of camera resolution as a reasonable limit
        self.display_width = int(self.cam_width * 0.8)
        self.display_height = int(self.cam_height * 0.8)

        # Data storage
        self.captured_images = []  # Will be restructured
        self.captured_landmarks = []  # Will be restructured
        self.labels = []

        # Constants for data structure
        self.MAX_HANDS = 2  # Maximum number of hands to detect
        self.NUM_LANDMARKS = 21  # MediaPipe hand has 21 landmarks
        self.NUM_COORDS = 3  # x, y, z coordinates
        self.CROP_SIZE = (112, 112)  # Resize dimension for cropped hands
        self.CROP_MARGIN = 40  # Increased margin for better hand context

        # Recording state variables
        self.is_recording = False
        self.no_hands_counter = 0
        self.no_hands_threshold = 10  # Stop recording after 10 frames with no hands
        self.frame_counter = 0
        self.last_hands_present = False
        self.auto_save_on_stop = True  # Always auto-save

        # Add a flag to track if data has been saved
        self.data_saved = True

        # UI Components
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMaximumSize(self.display_width, self.display_height)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Status indicator
        self.status_label = QLabel(
            "Enter glosses and show your hands to start recording"
        )
        self.status_label.setStyleSheet("color: blue; font-weight: bold;")
        self.status_label.setAlignment(Qt.AlignCenter)

        # Create a more visible section for controls
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)

        # Gloss input section with label
        gloss_layout = QHBoxLayout()
        gloss_label = QLabel("Enter glosses sequence:")
        self.gloss_input = QLineEdit()
        self.gloss_input.setPlaceholderText("Enter space-separated glosses...")
        self.gloss_input.setMinimumHeight(30)  # Make input field more visible
        gloss_layout.addWidget(gloss_label)
        gloss_layout.addWidget(self.gloss_input)

        # Clear glosses button
        self.clear_btn = QPushButton("Clear Glosses")
        self.clear_btn.clicked.connect(self.clear_glosses)
        gloss_layout.addWidget(self.clear_btn)

        # Add quit button to properly exit the application
        quit_layout = QHBoxLayout()
        self.quit_btn = QPushButton("Quit Application")
        self.quit_btn.setMinimumHeight(40)
        self.quit_btn.clicked.connect(self.close)
        self.quit_btn.setStyleSheet("background-color: #d9534f; color: white;")
        quit_layout.addWidget(self.quit_btn)

        # Add controls to their dedicated layout
        controls_layout.addLayout(gloss_layout)
        controls_layout.addLayout(quit_layout)  # Add the quit button layout
        controls_layout.addWidget(self.status_label)
        controls_layout.setSpacing(10)  # Add some spacing between controls

        # Main Layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_label)
        main_layout.addWidget(controls_widget)
        # Add more spacing between video and controls
        main_layout.setSpacing(15)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Timer to update frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # roughly 30 FPS

        # Variable to store the latest processed cropped hand image and landmarks
        self.latest_cropped = None
        self.latest_landmarks = None
        self.current_frame_has_hands = False

        # Set a reasonable window size - smaller since we have fewer controls
        self.resize(
            self.display_width, self.display_height + 120
        )  # Add space for controls

    def clear_glosses(self):
        """Clear the glosses input field."""
        self.gloss_input.clear()

    def start_recording(self):
        """Start recording automatically when hands appear."""
        glosses = self.get_glosses()
        if not glosses:
            self.status_label.setText("Please enter glosses before recording.")
            self.status_label.setStyleSheet("color: orange; font-weight: bold;")
            return False

        # Reset recording state
        self.is_recording = True
        self.frame_counter = 0
        self.no_hands_counter = 0
        self.captured_images = []
        self.captured_landmarks = []
        self.labels = glosses
        self.data_saved = False  # Reset saved flag

        # Update UI
        self.status_label.setText("Recording started. Capturing hand movements...")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        return True

    def stop_recording(self):
        """Stop recording when no hands are detected for a set period and auto-save."""
        self.is_recording = False

        frames_captured = len(self.captured_images)
        self.status_label.setText(
            f"Recording stopped. {frames_captured} frames captured."
        )
        self.status_label.setStyleSheet("color: green; font-weight: bold;")

        # Always auto-save if we have frames
        if frames_captured > 0:
            self.save_data()
        else:
            self.status_label.setText("Recording stopped. No frames captured.")
            self.status_label.setStyleSheet("color: orange; font-weight: bold;")
            self.data_saved = True  # Mark as saved since there's nothing to save

    def get_glosses(self):
        """Get the glosses from input field."""
        text = self.gloss_input.text().strip()
        return text.split() if text else []

    def update_frame(self):
        """Process camera frame and handle recording logic."""
        ret, frame = self.cap.read()
        if not ret:
            return

        # Flip the frame for natural interaction (mirror effect)
        frame = cv2.flip(frame, 1)

        # Create a deep copy for processing to avoid modifying the original
        process_frame = frame.copy()

        # Create a copy for annotations
        annotated_frame = frame.copy()

        # Process with MediaPipe
        process_frame.flags.writeable = False
        rgb_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(rgb_frame)
        process_frame.flags.writeable = True

        # Check if hands are present
        has_hands = bool(results.multi_hand_landmarks) and bool(
            results.multi_handedness
        )
        self.current_frame_has_hands = has_hands

        # Initialize per-frame data structures for landmarks and crops
        frame_landmarks = np.zeros(
            (self.MAX_HANDS, self.NUM_LANDMARKS, self.NUM_COORDS)
        )
        frame_crops = np.zeros((self.MAX_HANDS, *self.CROP_SIZE, 3), dtype=np.uint8)

        # Process hands if present
        if has_hands:
            # Process each detected hand
            for hand_idx, (hand_landmarks, handedness) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_handedness)
            ):
                if hand_idx >= self.MAX_HANDS:
                    break  # Skip if exceeding max hands

                # Get hand type (left or right) and confidence
                hand_type = handedness.classification[0].label
                hand_confidence = handedness.classification[0].score

                # Assign storage index based on handedness
                # Left hand goes to index 0, Right hand to index 1
                storage_idx = 0 if hand_type == "Left" else 1

                # Color for visualization
                color = (0, 0, 255) if hand_type == "Left" else (0, 255, 0)

                # Draw landmarks on annotated frame for visualization
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=color, thickness=2),
                )

                # Compute bounding box from landmarks with generous margin
                h, w, _ = frame.shape
                x_min, y_min, x_max, y_max = get_bounding_box(
                    hand_landmarks, w, h, margin=self.CROP_MARGIN
                )

                # Draw the bounding box on the annotated frame
                cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), color, 2)

                # Label the hand in the frame
                cv2.putText(
                    annotated_frame,
                    f"{hand_type} ({hand_confidence:.2f})",
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

                # Extract and process the hand region from the original clean frame
                if x_max > x_min and y_max > y_min:  # Valid bounding box
                    hand_img = frame[y_min:y_max, x_min:x_max].copy()
                    if hand_img.size > 0:
                        # Use better resizing method to maintain quality
                        cropped_resized = resize_preserve_aspect_ratio(
                            hand_img, self.CROP_SIZE
                        )
                        # Store in the position based on hand type
                        frame_crops[storage_idx] = cropped_resized

                # Extract landmark coordinates for this hand
                # for lm_idx, landmark in enumerate(hand_landmarks.landmark):
                #     # Store in the position based on hand type
                #     frame_landmarks[storage_idx, lm_idx] = [
                #         landmark.x,
                #         landmark.y,
                #         landmark.z,
                #     ]
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
                    frame_landmarks[storage_idx, idx] = skeletal[idx]

        # Auto-start recording if glosses are entered and hands appear
        glosses = self.get_glosses()
        if not self.is_recording and has_hands and glosses:
            self.start_recording()

        # Handle recording state
        if self.is_recording:
            # Add recording indicator to annotated frame
            cv2.putText(
                annotated_frame,
                f"Recording... Frame: {self.frame_counter}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

            # Check if hands are present
            if has_hands:
                # Reset counter if hands reappear
                self.no_hands_counter = 0

                # Store frame data (landmarks and crops)
                if frame_crops.any() and frame_landmarks.any():
                    self.captured_landmarks.append(frame_landmarks.copy())
                    self.captured_images.append(frame_crops.copy())
                    self.frame_counter += 1

                # Update last state
                self.last_hands_present = True

                # Update status
                self.status_label.setText(f"Recording... Frame: {self.frame_counter}")
            elif self.last_hands_present:  # Hands were present but now gone
                self.no_hands_counter += 1

                # Display countdown on annotated frame
                cv2.putText(
                    annotated_frame,
                    f"No hands! Recording will end in: {self.no_hands_threshold - self.no_hands_counter}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                self.status_label.setText(
                    f"No hands detected. Stopping in {self.no_hands_threshold - self.no_hands_counter} frames..."
                )

                # Auto-stop recording if no hands for threshold frames
                if self.no_hands_counter >= self.no_hands_threshold:
                    self.stop_recording()
        elif not has_hands:
            # Display waiting message on annotated frame
            cv2.putText(
                annotated_frame,
                "Waiting for hands to appear...",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        # Convert annotated frame to QImage and display in QLabel
        image = QImage(
            annotated_frame.data,
            annotated_frame.shape[1],
            annotated_frame.shape[0],
            annotated_frame.strides[0],
            QImage.Format_BGR888,
        )
        pixmap = QPixmap.fromImage(image)

        # Scale the pixmap while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

    def save_data(self):
        """Save captured data to an NPZ file without displaying popup notifications."""
        if not self.captured_images or len(self.captured_images) == 0:
            self.status_label.setText("No data to save.")
            self.status_label.setStyleSheet("color: orange; font-weight: bold;")
            self.data_saved = True  # Mark as saved since there's nothing to save
            return

        # Get glosses as a label
        glosses = self.get_glosses()
        if not glosses:
            glosses = ["Unlabeled"]

        # Create label part - sanitize to prevent path issues
        gloss_label = "-".join(
            [g.replace("/", "_").replace("\\", "_") for g in glosses]
        )

        # Generate Unix timestamp and encode in base64 for shorter filename
        unix_timestamp = int(time.time())
        timestamp = to_base64(unix_timestamp)

        # Create data directory if it doesn't exist
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data",
            "raw",
        )
        os.makedirs(data_dir, exist_ok=True)

        # Create output filename with base64 encoded timestamp
        output_file = os.path.join(data_dir, f"{gloss_label}_{timestamp}.npz")

        try:
            # Convert lists to numpy arrays with proper shapes
            # Landmarks shape: (num_frames, num_hands, num_landmarks, num_coords)
            skeletal_data = np.array(self.captured_landmarks)

            # Crops shape: (num_frames, num_hands, height, width, channels)
            crops = np.array(self.captured_images)

            optical = compute_optical_flow(crops)

            # Save with the required keys
            data = {
                "skeletal_data": skeletal_data,
                "crops": crops,
                "optical_flow": optical,
                "labels": np.array(glosses),
            }
            np.savez_compressed(output_file, **data)

            # Update status label only (no popup)
            self.status_label.setText(
                f"Data saved to {output_file} ({len(self.captured_images)} frames)"
            )
            self.status_label.setStyleSheet("color: green; font-weight: bold;")

            # Mark data as saved
            self.data_saved = True

            # Log to console for debugging
            print(f"Data saved silently to {output_file}")
            print(f"Shapes - Skeletal: {skeletal_data.shape}, Crops: {crops.shape}, Optical: {optical.shape}")

        except Exception as e:
            self.status_label.setText(f"Error saving data: {str(e)}")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            print(f"Error saving data: {str(e)}")

    def closeEvent(self, event):
        """Handle window close event."""
        # Check if recording is active and needs to be saved
        if self.is_recording:
            self.stop_recording()
            # Give a moment for the save operation to complete
            QApplication.processEvents()

        # Only save if there's unsaved data
        if len(self.captured_images) > 0 and not self.data_saved:
            self.save_data()

        self.cap.release()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignLanguageCapture()
    window.show()
    sys.exit(app.exec_())
