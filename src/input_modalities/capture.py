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
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import mediapipe as mp
import os
import time
import queue
import threading

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from utils.helpers import (
    get_bounding_box,
    resize_preserve_aspect_ratio,
    to_base64,
)
from input_modalities.optical_farneback import compute_optical_flow

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
mp_drawing = mp.solutions.drawing_utils

# Global constants - move outside class for better performance
MAX_HANDS = 2  # Maximum number of hands to detect
NUM_LANDMARKS = 21  # MediaPipe hand has 21 landmarks
NUM_COORDS = 3  # x, y, z coordinates
CROP_SIZE = (112, 112)  # Resize dimension for cropped hands
CROP_MARGIN = 40  # Margin for better hand context

# Frame buffer for producer-consumer pattern
FRAME_BUFFER_SIZE = 10

# Debug flag - set to False in production for better performance
DEBUG_DRAWING = False  # Controls whether to draw landmarks, bounding boxes, etc.


class FrameCapturingThread(QThread):
    """Thread for capturing frames from camera to decouple UI from frame capture."""

    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, camera_index=0, width=1920, height=1080):
        super().__init__()
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.running = False
        self.cap = None

    def run(self):
        """Main loop for capturing frames."""
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Flip the frame for mirror effect
                frame = cv2.flip(frame, 1)
                self.frame_ready.emit(frame)
            else:
                time.sleep(0.01)  # Short sleep to prevent CPU hogging

    def stop(self):
        """Stop the thread safely."""
        self.running = False
        if self.cap:
            self.cap.release()
        self.wait()


class DataSavingThread(QThread):
    """Thread for saving data in background without blocking the UI."""

    save_completed = pyqtSignal(bool, str)

    def __init__(self, data_to_save, output_file):
        super().__init__()
        self.data_to_save = data_to_save
        self.output_file = output_file

    def run(self):
        """Save the data to file in background."""
        try:
            np.savez_compressed(self.output_file, **self.data_to_save)
            self.save_completed.emit(True, f"Data saved to {self.output_file}")
            print(f"Data saved to {self.output_file}")
            print(
                f"Shapes - Skeletal: {self.data_to_save['skeletal_data'].shape}, "
                f"Crops: {self.data_to_save['crops'].shape}, "
                f"Optical: {self.data_to_save['optical_flow'].shape}"
            )
        except Exception as e:
            self.save_completed.emit(False, f"Error saving data: {str(e)}")
            print(f"Error saving data: {str(e)}")


class SignLanguageCapture(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Sign Language Capture")

        # Initialize capture thread
        self.capture_thread = FrameCapturingThread()
        self.capture_thread.frame_ready.connect(self.process_frame)
        self.capture_thread.start()

        # Get camera's native resolution
        self.cam_width = 1920
        self.cam_height = 1080

        # Calculate a reasonable video display size
        self.display_width = int(self.cam_width * 0.8)
        self.display_height = int(self.cam_height * 0.8)

        # Pre-allocate arrays for better performance
        # These will be reused rather than recreated each frame
        self.frame_landmarks = np.zeros((MAX_HANDS, NUM_LANDMARKS, NUM_COORDS))
        self.frame_crops = np.zeros((MAX_HANDS, *CROP_SIZE, 3), dtype=np.uint8)

        # Data storage
        self.captured_images = []
        self.captured_landmarks = []
        self.labels = []

        # Recording state variables
        self.is_recording = False
        self.no_hands_counter = 0
        self.no_hands_threshold = 10
        self.frame_counter = 0
        self.last_hands_present = False
        self.auto_save_on_stop = True
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
        self.gloss_input.setMinimumHeight(30)
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
        controls_layout.addLayout(quit_layout)
        controls_layout.addWidget(self.status_label)
        controls_layout.setSpacing(10)

        # Main Layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_label)
        main_layout.addWidget(controls_widget)
        main_layout.setSpacing(15)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Variable to store the latest processed cropped hand image and landmarks
        self.latest_cropped = None
        self.latest_landmarks = None
        self.current_frame_has_hands = False

        # Set a reasonable window size
        self.resize(self.display_width, self.display_height + 120)

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
            self.data_saved = True

    def get_glosses(self):
        """Get the glosses from input field."""
        text = self.gloss_input.text().strip()
        return text.split() if text else []

    def process_frame(self, frame):
        """Process received frame from capture thread."""
        # Create a deep copy for processing to avoid modifying the original
        process_frame = frame.copy()

        # Create a copy for annotations (only if needed)
        if DEBUG_DRAWING:
            annotated_frame = frame.copy()
        else:
            annotated_frame = frame  # Just use the original frame to save memory

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

        # Reset the arrays (faster than creating new ones)
        self.frame_landmarks.fill(0)
        self.frame_crops.fill(0)

        # Process hands if present
        if has_hands:
            # Process each detected hand
            for hand_idx, (hand_landmarks, handedness) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_handedness)
            ):
                if hand_idx >= MAX_HANDS:
                    break  # Skip if exceeding max hands

                # Get hand type (left or right) and confidence
                hand_type = handedness.classification[0].label
                hand_confidence = handedness.classification[0].score

                # Assign storage index based on handedness
                storage_idx = 0 if hand_type == "Left" else 1

                # Color for visualization (only compute if drawing)
                if DEBUG_DRAWING:
                    color = (0, 0, 255) if hand_type == "Left" else (0, 255, 0)
                    # Draw landmarks on annotated frame for visualization
                    mp_drawing.draw_landmarks(
                        annotated_frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(
                            color=color, thickness=2, circle_radius=2
                        ),
                        mp_drawing.DrawingSpec(color=color, thickness=2),
                    )

                # Compute bounding box from landmarks with generous margin
                h, w, _ = frame.shape
                x_min, y_min, x_max, y_max = get_bounding_box(
                    hand_landmarks, w, h, margin=CROP_MARGIN
                )

                # Draw the bounding box on the annotated frame (only if in debug mode)
                if DEBUG_DRAWING:
                    cv2.rectangle(
                        annotated_frame, (x_min, y_min), (x_max, y_max), color, 2
                    )
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
                            hand_img, CROP_SIZE
                        )
                        # Store in the position based on hand type
                        self.frame_crops[storage_idx] = cropped_resized

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
                    self.frame_landmarks[storage_idx, idx] = skeletal[idx]

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
                if self.frame_crops.any() and self.frame_landmarks.any():
                    self.captured_landmarks.append(self.frame_landmarks.copy())
                    self.captured_images.append(self.frame_crops.copy())
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
            skeletal_data = np.array(self.captured_landmarks)
            crops = np.array(self.captured_images)

            # Use threading for optical flow computation as it's CPU intensive
            self.status_label.setText("Processing optical flow data...")

            # Prepare the data for saving
            optical = compute_optical_flow(crops)

            # Prepare data dict
            data = {
                "skeletal_data": skeletal_data,
                "crops": crops,
                "optical_flow": optical,
                "labels": np.array(glosses),
            }

            # Update status
            self.status_label.setText("Saving data to file...")

            # Save in a separate thread to avoid blocking UI
            self.save_thread = DataSavingThread(data, output_file)
            self.save_thread.save_completed.connect(self.on_save_completed)
            self.save_thread.start()

        except Exception as e:
            self.status_label.setText(f"Error saving data: {str(e)}")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            print(f"Error saving data: {str(e)}")

    def on_save_completed(self, success, message):
        """Handle completion of the data saving operation."""
        if success:
            self.status_label.setText(message)
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
            self.data_saved = True
        else:
            self.status_label.setText(message)
            self.status_label.setStyleSheet("color: red; font-weight: bold;")

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
            # Wait briefly to allow save to start
            QApplication.processEvents()

        # Stop the capture thread
        self.capture_thread.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignLanguageCapture()
    window.show()
    sys.exit(app.exec_())
