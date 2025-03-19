import os
import json
import sys
import cv2
import numpy as np
import torch
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
import mediapipe as mp

from src.utils.helpers import get_bounding_box, resize_preserve_aspect_ratio
from src.models.model import CSLRModel
from src.input_modalities.optical_farneback import compute_optical_flow

class CSLRWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSLR Real-Time Prediction")
        self.setGeometry(100, 100, 800, 600)

        # Device and model setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = "checkpoints/best_model.pt"
        self.vocab = self.load_vocab("data/label-idx-mapping.json")
        self.vocab_size = len(self.vocab)
        self.idx_to_gloss = {idx: gloss for gloss, idx in self.vocab.items()}
        self.model = self.load_model()

        # MediaPipe setup
        self.mp_hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils

        # Video capture initialization
        self.cap = None
        self.initialize_camera()

        # Recording state variables
        self.is_recording = False
        self.no_hands_counter = 0
        self.no_hands_threshold = 10
        self.min_sequence_length = 10
        self.max_sequence_length = 100

        # Constants for pre-allocation
        self.MAX_HANDS = 2
        self.NUM_LANDMARKS = 21
        self.NUM_COORDS = 3
        self.CROP_SIZE = (112, 112)

        # Dynamic buffers
        self.skeletal_buffer = []
        self.crops_buffer = []
        self.optical_flow_buffer = []

        # GUI components
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.prediction_label = QLabel("Prediction: Waiting for hands...", self)
        self.start_button = QPushButton("Start", self)
        self.stop_button = QPushButton("Stop", self)

        # Layout
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

        # Timer for video feed
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def initialize_camera(self):
        camera_indices = [0, 1, 2]
        for index in camera_indices:
            self.cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
            if self.cap.isOpened():
                break
            self.cap.release()
            self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if self.cap.isOpened():
                break
            self.cap.release()

        if not self.cap or not self.cap.isOpened():
            self.prediction_label.setText("Error: Could not open camera")
            self.start_button.setEnabled(False)
            return

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera opened at index {index} with resolution {self.frame_width}x{self.frame_height}")

    def load_vocab(self, json_path):
        try:
            with open(json_path, 'r') as f:
                vocab = json.load(f)
            return vocab
        except Exception as e:
            print(f"Error loading vocab from JSON: {e}")
            raise

    def load_model(self):
        spatial_params = {"D_spatial": 128}
        temporal_params = {
            "in_channels": 128,
            "out_channels": 256,
            "kernel_sizes": [3, 5, 7],
            "dilations": [1, 2, 4],
            "vocab_size": self.vocab_size
        }
        transformer_params = {
            "input_dim": 2 * 256,
            "model_dim": 256,
            "num_heads": 4,
            "num_layers": 2,
            "vocab_size": self.vocab_size,
            "dropout": 0.1
        }
        enstim_params = {
            "vocab_size": self.vocab_size,
            "context_dim": 256,
            "blank": 0,
            "lambda_entropy": 0.1
        }
        model = CSLRModel(spatial_params, temporal_params, transformer_params, enstim_params, device=self.device).to(self.device)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        model.load_state_dict(checkpoint)
        model.eval()
        return model

    def start_capture(self):
        if not self.cap or not self.cap.isOpened():
            self.prediction_label.setText("Error: Camera not available")
            return
        self.timer.start(30)
        self.prediction_label.setText("Prediction: Waiting for hands...")

    def stop_capture(self):
        self.timer.stop()
        self.is_recording = False
        self.no_hands_counter = 0
        self.skeletal_buffer.clear()
        self.crops_buffer.clear()
        self.optical_flow_buffer.clear()
        self.prediction_label.setText("Prediction: Stopped")

    def update_frame(self):
        if not self.cap or not self.cap.isOpened():
            self.prediction_label.setText("Error: Camera not available")
            self.timer.stop()
            return

        ret, frame = self.cap.read()
        if not ret:
            self.prediction_label.setText("Error: Failed to capture frame")
            return

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(frame_rgb)
        has_hands = bool(results.multi_hand_landmarks)

        # Process frame data
        skeletal_data, crops_data = self.process_frame_data(frame, results)

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

        # Display the frame with annotations
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                x_min, y_min, x_max, y_max = get_bounding_box(hand_landmarks, self.frame_width, self.frame_height)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        frame_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_display.shape
        bytes_per_line = ch * w
        q_image = QImage(frame_display.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

    def start_recording(self):
        self.is_recording = True
        self.skeletal_buffer = []
        self.crops_buffer = []
        self.optical_flow_buffer = []
        self.prediction_label.setText("Recording...")

    def append_to_buffers(self, skeletal_data, crops_data):
        self.skeletal_buffer.append(skeletal_data)
        self.crops_buffer.append(crops_data)
        if len(self.skeletal_buffer) > self.max_sequence_length:
            self.skeletal_buffer.pop(0)
            self.crops_buffer.pop(0)

    def stop_recording_and_predict(self):
        self.is_recording = False
        sequence_length = len(self.skeletal_buffer)
        if sequence_length >= self.min_sequence_length:
            skeletal_array = np.array(self.skeletal_buffer)  # (T, 2, 21, 3)
            crops_array = np.array(self.crops_buffer, dtype=np.uint8)  # (T, 2, 112, 112, 3)

            # Compute optical flow from crops
            optical_flow_array = compute_optical_flow(crops_array)  # (T-1, 2, 2, 112, 112)

            # Pad optical flow to T frames
            T = skeletal_array.shape[0]
            if optical_flow_array.shape[0] == T - 1:
                zero_flow = np.zeros((1, 2, 112, 112, 2), dtype=optical_flow_array.dtype)
                optical_flow_padded = np.concatenate([zero_flow, optical_flow_array], axis=0)  # (T, 2, 112, 112, 2)
            else:
                optical_flow_padded = optical_flow_array

            # Convert to tensors
            skeletal_tensor = torch.tensor(skeletal_array, dtype=torch.float32).to(self.device)  # (T, 2, 21, 3)
            crops_tensor = torch.tensor(crops_array / 255.0, dtype=torch.float32).permute(0, 1, 4, 2, 3).to(self.device)  # (T, 2, 3, 112, 112)
            optical_flow_tensor = torch.tensor(optical_flow_padded, dtype=torch.float32).to(self.device)  # (T, 2, 2, 112, 112)
            input_lengths = torch.tensor([sequence_length], dtype=torch.long).to(self.device)

            # Add batch dimension
            skeletal_tensor = skeletal_tensor.unsqueeze(0)  # (1, T, 2, 21, 3)
            crops_tensor = crops_tensor.unsqueeze(0)  # (1, T, 2, 3, 112, 112)
            optical_flow_tensor = optical_flow_tensor.unsqueeze(0)  # (1, T, 2, 2, 112, 112)

            # Model inference
            with torch.no_grad():
                pred_sequences = self.model.decode(skeletal_tensor, crops_tensor, optical_flow_tensor, input_lengths)
                pred_glosses = ' '.join([self.idx_to_gloss[idx] for idx in pred_sequences[0]])
                self.prediction_label.setText(f"Prediction: {pred_glosses}")
        else:
            self.prediction_label.setText("Prediction: Sequence too short")
        self.skeletal_buffer.clear()
        self.crops_buffer.clear()
        self.optical_flow_buffer.clear()

    def process_frame_data(self, frame, results):
        # Pre-allocate arrays for 2 hands
        skeletal_data = np.zeros((self.MAX_HANDS, self.NUM_LANDMARKS, self.NUM_COORDS), dtype=np.float32)
        crops_data = np.zeros((self.MAX_HANDS, *self.CROP_SIZE, 3), dtype=np.uint8)

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks[:self.MAX_HANDS]):  # Limit to MAX_HANDS
                # Skeletal data
                landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                normalized_landmarks = self.normalize_landmarks(landmarks)
                skeletal_data[idx] = normalized_landmarks  # Assign to pre-allocated slot

                # Crop data
                x_min, y_min, x_max, y_max = get_bounding_box(hand_landmarks, self.frame_width, self.frame_height)
                if x_max > x_min and y_max > y_min:
                    crop = frame[y_min:y_max, x_min:x_max]  # BGR, uint8
                    crop_resized = resize_preserve_aspect_ratio(crop, self.CROP_SIZE)
                    crops_data[idx] = crop_resized  # Assign to pre-allocated slot

        return skeletal_data, crops_data

    def normalize_landmarks(self, landmarks):
        wrist = landmarks[0]
        translated = [[lm[0] - wrist[0], lm[1] - wrist[1], lm[2] - wrist[2]] for lm in landmarks]
        ref_vec = [translated[9][i] for i in range(3)]
        ref_length = np.sqrt(sum(v**2 for v in ref_vec))
        if ref_length > 0:
            normalized = [[lm[i] / ref_length for i in range(3)] for lm in translated]
        else:
            normalized = translated
        return np.array(normalized, dtype=np.float32)  # Convert to array for consistency

    def closeEvent(self, event):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.mp_hands.close()
        self.timer.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CSLRWindow()
    window.show()
    sys.exit(app.exec_())