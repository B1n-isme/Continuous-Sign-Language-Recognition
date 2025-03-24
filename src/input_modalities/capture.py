import os
import sys
import cv2
import numpy as np
import queue
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
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import mediapipe as mp
import time

# Custom utilities and optical flow
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from utils.helpers import get_bounding_box, resize_preserve_aspect_ratio, to_base64
from input_modalities.optical_farneback import compute_optical_flow
from src.utils.config_loader import load_config

# Initialize MediaPipe Hands (global to avoid re-instantiation)
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
mp_drawing = mp.solutions.drawing_utils

# Load configuration
config = load_config("configs/data_config.yaml")

# Data directory
data_dir = config["paths"]["raw_data"]
os.makedirs(data_dir, exist_ok=True)

# Global constants
MAX_HANDS = config["hand_landmarks"]["max_hands"]
NUM_LANDMARKS = config["hand_landmarks"]["num_landmarks"]
NUM_COORDS = config["hand_landmarks"]["num_coords"]
CROP_SIZE = config["crops"]["crop_size"]
CROP_MARGIN = config["crops"]["crop_margin"]

# Debug flag (set False in production)
DEBUG_DRAWING = False

# -------------------- Frame Capture Thread --------------------
class FrameCapturingThread(QThread):
    def __init__(self, frame_queue, camera_index=0, width=1920, height=1080):
        super().__init__()
        self.frame_queue = frame_queue
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.running = False
        self.cap = None

    def run(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Flip for mirror effect
                frame = cv2.flip(frame, 1)
                # Try to push the frame into the queue; if full, drop the old frame.
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.frame_queue.put_nowait(frame)
            else:
                time.sleep(0.001)  # Short sleep to reduce CPU usage

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.wait()

# -------------------- Frame Processing Thread --------------------
class FrameProcessingThread(QThread):
    # Emit the annotated frame, normalized landmarks, cropped images, and hand-presence flag
    frame_processed = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, bool)

    def __init__(self, frame_queue):
        super().__init__()
        self.frame_queue = frame_queue
        self.running = False

    def run(self):
        self.running = True
        while self.running:
            try:
                # Get the latest frame (dropping older frames)
                frame = self.frame_queue.get(timeout=0.005)
            except queue.Empty:
                continue
            annotated, landmarks_array, crops_array, has_hands = self.process_frame(frame)
            self.frame_processed.emit(annotated, landmarks_array, crops_array, has_hands)

    def process_frame(self, frame):
        # Create copies for processing and (if needed) annotation
        proc_frame = frame.copy()
        annotated = frame.copy() if DEBUG_DRAWING else frame

        proc_frame.flags.writeable = False
        rgb_frame = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(rgb_frame)
        proc_frame.flags.writeable = True

        has_hands = (results.multi_hand_landmarks is not None and 
                     results.multi_handedness is not None)

        # Pre-allocate arrays for landmarks and crops
        landmarks_array = np.zeros((MAX_HANDS, NUM_LANDMARKS, NUM_COORDS), dtype=np.float32)
        crops_array = np.zeros((MAX_HANDS, *CROP_SIZE, 3), dtype=np.uint8)

        if has_hands:
            for hand_idx, (hand_landmarks, handedness) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_handedness)
            ):
                if hand_idx >= MAX_HANDS:
                    break

                hand_type = handedness.classification[0].label
                storage_idx = 0 if hand_type == "Left" else 1
                color = (0, 0, 255) if hand_type == "Left" else (0, 255, 0)

                if DEBUG_DRAWING:
                    mp_drawing.draw_landmarks(
                        annotated,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=color, thickness=2),
                    )

                h, w, _ = frame.shape
                x_min, y_min, x_max, y_max = get_bounding_box(hand_landmarks, w, h, margin=CROP_MARGIN)

                if DEBUG_DRAWING:
                    cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), color, 2)
                    cv2.putText(annotated, f"{hand_type}", (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if x_max > x_min and y_max > y_min:
                    hand_img = frame[y_min:y_max, x_min:x_max]
                    if hand_img.size:
                        resized = resize_preserve_aspect_ratio(hand_img.copy(), CROP_SIZE)
                        crops_array[storage_idx] = resized

                # Vectorized normalization of landmarks:
                landmarks = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32
                )
                landmarks -= landmarks[0]  # Translate relative to the wrist
                ref_length = np.linalg.norm(landmarks[9])  # Use middle finger MCP as reference
                if ref_length > 0:
                    landmarks /= ref_length
                z_max = np.max(np.abs(landmarks[:, 2]))
                if z_max > 0:
                    landmarks[:, 2] /= z_max
                landmarks_array[storage_idx] = landmarks

        return annotated, landmarks_array, crops_array, has_hands

    def stop(self):
        self.running = False
        self.wait()

# -------------------- Data Saving Thread --------------------
class DataSavingThread(QThread):
    save_completed = pyqtSignal(bool, str)

    def __init__(self, data_to_save, output_file):
        super().__init__()
        self.data_to_save = data_to_save
        self.output_file = output_file

    def run(self):
        try:
            if "optical_flow" not in self.data_to_save:
                self.data_to_save["optical_flow"] = compute_optical_flow(self.data_to_save["crops"])
            np.savez_compressed(self.output_file, **self.data_to_save)
            self.save_completed.emit(True, f"Data saved to {self.output_file}")
            print(f"Data saved to {self.output_file}")
        except Exception as e:
            self.save_completed.emit(False, f"Error saving data: {str(e)}")
            print(f"Error saving data: {str(e)}")

# -------------------- Main Application --------------------
class SignLanguageCapture(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Sign Language Capture")

        # Create a shared queue for frames (size=1 to always hold the latest frame)
        self.frame_queue = queue.Queue(maxsize=1)

        # Start the capture and processing threads
        self.capture_thread = FrameCapturingThread(self.frame_queue)
        self.processing_thread = FrameProcessingThread(self.frame_queue)
        self.capture_thread.start()
        self.processing_thread.frame_processed.connect(self.on_frame_processed)
        self.processing_thread.start()

        # Camera and display settings
        self.cam_width = 1920
        self.cam_height = 1080
        self.display_width = int(self.cam_width * 0.8)
        self.display_height = int(self.cam_height * 0.8)

        # Data storage for recording
        self.captured_images = []
        self.captured_landmarks = []
        self.labels = []

        # Recording state
        self.is_recording = False
        self.no_hands_counter = 0
        self.no_hands_threshold = 10
        self.frame_counter = 0
        self.last_hands_present = False
        self.data_saved = True

        # UI Components
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMaximumSize(self.display_width, self.display_height)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.status_label = QLabel("Enter glosses and show your hands to start recording")
        self.status_label.setStyleSheet("color: blue; font-weight: bold;")
        self.status_label.setAlignment(Qt.AlignCenter)

        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        gloss_layout = QHBoxLayout()
        gloss_label = QLabel("Enter glosses sequence:")
        self.gloss_input = QLineEdit()
        self.gloss_input.setPlaceholderText("Enter space-separated glosses...")
        self.gloss_input.setMinimumHeight(30)
        gloss_layout.addWidget(gloss_label)
        gloss_layout.addWidget(self.gloss_input)
        self.clear_btn = QPushButton("Clear Glosses")
        self.clear_btn.clicked.connect(self.clear_glosses)
        gloss_layout.addWidget(self.clear_btn)
        quit_layout = QHBoxLayout()
        self.quit_btn = QPushButton("Quit Application")
        self.quit_btn.setMinimumHeight(40)
        self.quit_btn.clicked.connect(self.close)
        self.quit_btn.setStyleSheet("background-color: #d9534f; color: white;")
        quit_layout.addWidget(self.quit_btn)
        controls_layout.addLayout(gloss_layout)
        controls_layout.addLayout(quit_layout)
        controls_layout.addWidget(self.status_label)
        controls_layout.setSpacing(10)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_label)
        main_layout.addWidget(controls_widget)
        main_layout.setSpacing(15)
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        self.resize(self.display_width, self.display_height + 120)

    def clear_glosses(self):
        self.gloss_input.clear()

    def start_recording(self):
        glosses = self.get_glosses()
        if not glosses:
            self.status_label.setText("Please enter glosses before recording.")
            self.status_label.setStyleSheet("color: orange; font-weight: bold;")
            return False
        self.is_recording = True
        self.frame_counter = 0
        self.no_hands_counter = 0
        self.captured_images.clear()
        self.captured_landmarks.clear()
        self.labels = glosses
        self.data_saved = False
        self.status_label.setText("Recording started. Capturing hand movements...")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        return True

    def stop_recording(self):
        self.is_recording = False
        frames_captured = len(self.captured_images)
        self.status_label.setText(f"Recording stopped. {frames_captured} frames captured.")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        if frames_captured > 0:
            self.save_data()
        else:
            self.status_label.setText("Recording stopped. No frames captured.")
            self.status_label.setStyleSheet("color: orange; font-weight: bold;")
            self.data_saved = True

    def get_glosses(self):
        text = self.gloss_input.text().strip()
        return text.split() if text else []

    def on_frame_processed(self, annotated_frame, landmarks_array, crops_array, has_hands):
        # Auto-start recording if glosses are provided and hands appear
        if not self.is_recording and has_hands and self.get_glosses():
            self.start_recording()

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
            if has_hands:
                self.no_hands_counter = 0
                self.captured_landmarks.append(landmarks_array.copy())
                self.captured_images.append(crops_array.copy())
                self.frame_counter += 1
                self.last_hands_present = True
                self.status_label.setText(f"Recording... Frame: {self.frame_counter}")
            elif self.last_hands_present:
                self.no_hands_counter += 1
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
                if self.no_hands_counter >= self.no_hands_threshold:
                    self.stop_recording()
        else:
            if not has_hands:
                cv2.putText(
                    annotated_frame,
                    "Waiting for hands to appear...",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

        # Convert annotated frame to QImage and update the display
        image = QImage(
            annotated_frame.data,
            annotated_frame.shape[1],
            annotated_frame.shape[0],
            annotated_frame.strides[0],
            QImage.Format_BGR888,
        )
        pixmap = QPixmap.fromImage(image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

    def save_data(self):
        if not self.captured_images:
            self.status_label.setText("No data to save.")
            self.status_label.setStyleSheet("color: orange; font-weight: bold;")
            self.data_saved = True
            return

        glosses = self.get_glosses() or ["Unlabeled"]
        gloss_label = "-".join(g.replace("/", "_").replace("\\", "_") for g in glosses)
        unix_timestamp = int(time.time())
        timestamp = to_base64(unix_timestamp)
        # data_dir = os.path.join(
        #     os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        #     "data",
        #     "raw",
        # )
        output_file = os.path.join(data_dir, f"{gloss_label}_{timestamp}.npz")

        try:
            skeletal_data = np.array(self.captured_landmarks)
            crops = np.array(self.captured_images)
            data = {
                "skeletal_data": skeletal_data,
                "crops": crops,
                "labels": np.array(glosses),
            }
            self.status_label.setText("Saving data to file...")
            self.save_thread = DataSavingThread(data, output_file)
            self.save_thread.save_completed.connect(self.on_save_completed)
            self.save_thread.start()
        except Exception as e:
            self.status_label.setText(f"Error saving data: {str(e)}")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            print(f"Error saving data: {str(e)}")

    def on_save_completed(self, success, message):
        if success:
            self.status_label.setText(message)
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
            self.data_saved = True
        else:
            self.status_label.setText(message)
            self.status_label.setStyleSheet("color: red; font-weight: bold;")

    def closeEvent(self, event):
        if self.is_recording:
            self.stop_recording()
            QApplication.processEvents()
        if self.captured_images and not self.data_saved:
            self.save_data()
            QApplication.processEvents()
        self.capture_thread.stop()
        self.processing_thread.stop()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignLanguageCapture()
    window.show()
    sys.exit(app.exec_())
