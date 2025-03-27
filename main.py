import os
import json
import sys
import cv2
import numpy as np
import torch
from torch.amp import autocast, GradScaler
import queue
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import mediapipe as mp

from src.utils.helpers import get_bounding_box, resize_preserve_aspect_ratio
from src.feature_extraction.skeletal_aug import interpolate_skeletal, smooth_skeletal_ema
from src.feature_extraction.crop_aug import normalize_crops
from src.feature_extraction.flow_aug import normalize_optical_flow
from src.models.model import CSLRModel
from src.input_modalities.optical_farneback import compute_optical_flow
from src.models.ema import EMA, get_decay

# -------------------- Capture & Processing Threads --------------------

class CaptureThread(QThread):
    """Continuously capture frames from the camera and push them into a queue."""
    def __init__(self, cap, frame_queue, width, height):
        super().__init__()
        self.cap = cap
        self.frame_queue = frame_queue
        self.width = width
        self.height = height
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Mirror effect
                frame = cv2.flip(frame, 1)
                # Always keep only the latest frame in the queue
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.frame_queue.put_nowait(frame)
            else:
                self.msleep(1)  # minimal sleep to reduce CPU load

    def stop(self):
        self.running = False
        self.wait()


class ProcessingThread(QThread):
    """
    Continuously pull the latest frame from the queue, process it
    with MediaPipe and compute skeletal and crop data.
    """
    frame_processed = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, bool)

    def __init__(self, frame_queue, frame_width, frame_height, crop_size, max_hands=2):
        super().__init__()
        self.frame_queue = frame_queue
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.crop_size = crop_size
        self.max_hands = max_hands
        self.running = True

        # Initialize MediaPipe Hands once (reuse for performance)
        self.mp_hands = mp.solutions.hands.Hands(
            max_num_hands=2, min_detection_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.CROP_MARGIN = 40  # as in original

    def run(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.005)
            except queue.Empty:
                continue
            annotated, skeletal_data, crops_data, has_hands = self.process_frame(frame)
            self.frame_processed.emit(annotated, skeletal_data, crops_data, has_hands)

    def process_frame(self, frame):
        # Create a copy for processing and (if needed) annotation
        proc_frame = frame.copy()
        # If not debugging, we can use the same frame to save memory
        annotated = proc_frame.copy()
        
        proc_frame.flags.writeable = False
        rgb_frame = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(rgb_frame)
        proc_frame.flags.writeable = True

        has_hands = results.multi_hand_landmarks is not None

        # Pre-allocate arrays for up to 2 hands
        NUM_LANDMARKS = 21
        NUM_COORDS = 3
        skeletal_data = np.zeros((self.max_hands, NUM_LANDMARKS, NUM_COORDS), dtype=np.float32)
        crops_data = np.zeros((self.max_hands, *self.crop_size, 3), dtype=np.float32)

        if has_hands:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks[: self.max_hands]):
                # Determine hand type for ordering (Left=slot 0, Right=slot 1)
                hand_type = results.multi_handedness[idx].classification[0].label
                storage_idx = 0 if hand_type == "Left" else 1
                color = (0, 0, 255) if hand_type == "Left" else (0, 255, 0)
                
                # Optionally draw landmarks if needed (e.g. for debug)
                self.mp_drawing.draw_landmarks(annotated, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                # Compute bounding box with margin
                x_min, y_min, x_max, y_max = get_bounding_box(hand_landmarks, self.frame_width, self.frame_height, margin=self.CROP_MARGIN)
                # Draw box for visualization if desired
                cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(annotated, f"{hand_type}", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Crop and resize the hand region
                if x_max > x_min and y_max > y_min:
                    hand_img = frame[y_min:y_max, x_min:x_max]
                    if hand_img.size:
                        crop_resized = resize_preserve_aspect_ratio(hand_img.copy(), self.crop_size)
                        crops_data[storage_idx] = crop_resized

                # Vectorized normalization of landmarks
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
                # Translate landmarks so wrist (index 0) is the origin
                landmarks -= landmarks[0]
                # Scale normalization using distance from wrist to middle finger MCP (index 9)
                ref_length = np.linalg.norm(landmarks[9])
                if ref_length > 0:
                    landmarks /= ref_length
                # Optional depth normalization
                z_max = np.max(np.abs(landmarks[:, 2]))
                if z_max > 0:
                    landmarks[:, 2] /= z_max
                skeletal_data[storage_idx] = landmarks

        return annotated, skeletal_data, crops_data, has_hands

    def stop(self):
        self.running = False
        self.wait()

# -------------------- Main Application --------------------

class CSLRWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSLR Real-Time Prediction")
        self.setGeometry(100, 100, 800, 600)

        # Device and model setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = "models/checkpoints/best_model.pt"
        self.vocab = self.load_vocab("data/label-idx-mapping.json")
        self.vocab_size = len(self.vocab)
        self.idx_to_gloss = {idx: gloss for gloss, idx in self.vocab.items()}
        self.model = self.load_model()

        # MediaPipe will be used in the processing thread

        # Recording state variables and buffers
        self.is_recording = False
        self.no_hands_counter = 0
        self.no_hands_threshold = 10
        self.min_sequence_length = 10
        self.max_sequence_length = 100
        self.skeletal_buffer = []
        self.crops_buffer = []

        # Constants for pre-allocation in processing thread
        self.MAX_HANDS = 2
        self.CROP_SIZE = (224, 224)

        # GUI components
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.prediction_label = QLabel("Prediction: Waiting for hands...", self)
        self.start_button = QPushButton("Start", self)
        self.stop_button = QPushButton("Stop", self)

        # Layout setup
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.prediction_label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Button connections
        self.start_button.clicked.connect(self.start_capture)
        self.stop_button.clicked.connect(self.stop_capture)

        # Camera initialization
        self.cap = None
        self.initialize_camera()
        if not (self.cap and self.cap.isOpened()):
            return
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera initialized: {self.frame_width}x{self.frame_height}")

        # Create a queue to hold only the latest frame
        self.frame_queue = queue.Queue(maxsize=1)
        # Start capture and processing threads
        self.capture_thread = CaptureThread(self.cap, self.frame_queue, self.frame_width, self.frame_height)
        self.processing_thread = ProcessingThread(self.frame_queue, self.frame_width, self.frame_height, self.CROP_SIZE)
        self.processing_thread.frame_processed.connect(self.on_frame_processed)
        self.capture_thread.start()
        self.processing_thread.start()

    def initialize_camera(self):
        """Try several camera indices to initialize the video capture."""
        camera_indices = [0, 1, 2, 3]
        for index in camera_indices:
            print(f"Attempting to open camera at index {index}...")
            cap = cv2.VideoCapture(index)
            if cap is not None and cap.isOpened():
                self.cap = cap
                print(f"Camera opened at index {index}")
                return
            if cap:
                cap.release()
        print("Failed to open any camera")
        self.prediction_label.setText("Error: Could not open camera")
        self.start_button.setEnabled(False)

    def load_vocab(self, json_path):
        try:
            with open(json_path, "r") as f:
                vocab = json.load(f)
            return vocab
        except Exception as e:
            print(f"Error loading vocab: {e}")
            raise

    def load_model(self):
        spatial_params = {"D_spatial": 128}
        temporal_params = {
            "in_channels": 128,
            "out_channels": 256,
            "kernel_sizes": [3, 5, 7],
            "dilations": [1, 2, 4],
            "vocab_size": self.vocab_size,
        }
        transformer_params = {
            "input_dim": 2 * 256,
            "model_dim": 256,
            "num_heads": 4,
            "num_layers": 2,
            "vocab_size": self.vocab_size,
            "dropout": 0.1,
        }
        enstim_params = {
            "vocab_size": self.vocab_size,
            "context_dim": 256,
            "blank": 0,
            "lambda_entropy": 0.1,
        }
        model = CSLRModel(
            spatial_params,
            temporal_params,
            transformer_params,
            enstim_params,
            label_mapping_path="data/label-idx-mapping.json",
            device=self.device,
        ).to(self.device)
        # Load checkpoint
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            # Load EMA if available
            if 'ema_shadow' in checkpoint:
                ema = EMA(model)
                ema.shadow = checkpoint['ema_shadow']
                ema.apply_shadow()
                if 'ema_decay' in checkpoint:
                    ema.decay = checkpoint['ema_decay']
        except Exception as e:
            print(f"Failed to load model checkpoint: {e}")
            raise
        model.eval()
        return model

    def start_capture(self):
        if not (self.cap and self.cap.isOpened()):
            self.prediction_label.setText("Error: Camera not available")
            return
        self.prediction_label.setText("Prediction: Waiting for hands...")

    def stop_capture(self):
        # Stop recording buffers (video feed threads remain running)
        self.is_recording = False
        self.no_hands_counter = 0
        self.skeletal_buffer.clear()
        self.crops_buffer.clear()
        self.prediction_label.setText("Prediction: Stopped")

    def on_frame_processed(self, annotated_frame, skeletal_data, crops_data, has_hands):
        # Update the video display
        image = QImage(
            annotated_frame.data,
            annotated_frame.shape[1],
            annotated_frame.shape[0],
            annotated_frame.strides[0],
            QImage.Format_BGR888,
        )
        self.video_label.setPixmap(QPixmap.fromImage(image))

        # Dynamic recording logic
        if has_hands:
            if not self.is_recording:
                self.start_recording()
            self.append_to_buffers(skeletal_data, crops_data)
            self.no_hands_counter = 0
        else:
            if self.is_recording:
                self.no_hands_counter += 1
                if self.no_hands_counter >= self.no_hands_threshold:
                    self.stop_recording_and_predict()

    def start_recording(self):
        self.is_recording = True
        self.skeletal_buffer.clear()
        self.crops_buffer.clear()
        self.prediction_label.setText("Recording...")

    def append_to_buffers(self, skeletal_data, crops_data):
        self.skeletal_buffer.append(skeletal_data)
        self.crops_buffer.append(crops_data)
        # Limit buffer to max sequence length
        if len(self.skeletal_buffer) > self.max_sequence_length:
            self.skeletal_buffer.pop(0)
            self.crops_buffer.pop(0)

    def stop_recording_and_predict(self):
        self.is_recording = False
        sequence_length = len(self.skeletal_buffer)
        if sequence_length >= self.min_sequence_length:
            skeletal_array = np.array(self.skeletal_buffer)  # (T, 2, 21, 3)
            crops_array = np.array(self.crops_buffer, dtype=np.float32)  # (T, 2, 112, 112, 3)

            # Compute optical flow from crops (assumed to be CPU-intensive)
            optical_flow_array = compute_optical_flow(crops_array)  # (T-1, 2, 2, 112, 112)
            T = skeletal_array.shape[0]
            if optical_flow_array.shape[0] == T - 1:
                zero_flow = np.zeros((1, 2, 224, 224, 2), dtype=optical_flow_array.dtype)
                optical_flow_padded = np.concatenate([zero_flow, optical_flow_array], axis=0)
            else:
                optical_flow_padded = optical_flow_array

            # --- Normalization / Preprocessing Steps for Inference ---
            # For skeletal data: apply deterministic cleaning (interpolation and smoothing)
            skeletal_normalized = smooth_skeletal_ema(interpolate_skeletal(skeletal_array))

            # For cropped images: normalize pixel values to [0, 1]
            crops_normalized = normalize_crops(crops_array)

            # For optical flow: normalize the flow vectors
            optical_flow_normalized = normalize_optical_flow(optical_flow_padded)

            # Convert to tensors and add batch dimension
            skeletal_tensor = torch.tensor(skeletal_normalized, dtype=torch.float32).to(self.device).unsqueeze(0)  # (1, T, 2, 21, 3)
            crops_tensor = (
                torch.tensor(crops_normalized / 255.0, dtype=torch.float32)
                .permute(0, 1, 4, 2, 3)
                .to(self.device)
                .unsqueeze(0)  # (1, T, 2, 3, 112, 112)
            )
            optical_flow_tensor = torch.tensor(optical_flow_normalized, dtype=torch.float32).to(self.device).unsqueeze(0)  # (1, T, 2, 2, 112, 112)
            input_lengths = torch.tensor([sequence_length], dtype=torch.long).to(self.device)

            lm_path = "models/checkpoints/kenlm.binary" 

            # Model inference
            with torch.no_grad():
                with autocast(device_type="cpu"):
                    pred_sequences = self.model.decode(skeletal_tensor, crops_tensor, optical_flow_tensor, input_lengths)
                    # pred_sequences = self.model.decode_with_lm(skeletal_tensor, crops_tensor, optical_flow_tensor, input_lengths, lm_path=lm_path, beam_size=10, lm_weight=0.5)
                    # pred_glosses = " ".join([self.idx_to_gloss[idx] for idx in pred_sequences[0]])
                pred_glosses = " ".join(pred_sequences[0])
                self.prediction_label.setText(f"Prediction: {pred_glosses}")
        else:
            self.prediction_label.setText("Prediction: Sequence too short")
        self.skeletal_buffer.clear()
        self.crops_buffer.clear()

    def closeEvent(self, event):
        # Stop threads and release resources
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.processing_thread.stop()
        self.capture_thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CSLRWindow()
    window.show()
    sys.exit(app.exec_())
