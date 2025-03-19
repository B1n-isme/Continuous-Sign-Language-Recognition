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

from src.utils.helpers import (
    get_bounding_box,
    resize_preserve_aspect_ratio
)
from src.models.model import CSLRModel  # Adjust this import based on your project structure

class CSLRWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSLR Real-Time Prediction")
        self.setGeometry(100, 100, 800, 600)

        # Device and model setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = "checkpoints/best_model.pt"  # Adjust path
        self.vocab = self.load_vocab("data/label-idx-mapping.json")  # Use JSON file
        self.vocab_size = len(self.vocab)
        self.idx_to_gloss = {idx: gloss for gloss, idx in self.vocab.items()}
        self.model = self.load_model()

        # MediaPipe setup
        self.mp_hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils

        # Video capture
        self.cap = cv2.VideoCapture(0)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Buffer for sequence data
        self.sequence_length = 30  # Frames per sequence
        self.skeletal_buffer = []
        self.crops_buffer = []
        self.optical_flow_buffer = []
        self.prev_gray = None

        # GUI components
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.prediction_label = QLabel("Prediction: ", self)
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

    def load_vocab(self, json_path):
        """Load vocabulary from a JSON file."""
        try:
            with open(json_path, 'r') as f:
                vocab = json.load(f)
            print("Loaded vocabulary:", vocab)  # Debug print
            return vocab
        except Exception as e:
            print(f"Error loading vocab from JSON: {e}")
            raise

    def load_model(self):
        """Load the trained CSLRModel from checkpoint."""
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
        """Start the video capture and processing."""
        self.timer.start(30)  # ~33ms interval for ~30 FPS

    def stop_capture(self):
        """Stop the video capture and clear buffers."""
        self.timer.stop()
        self.skeletal_buffer.clear()
        self.crops_buffer.clear()
        self.optical_flow_buffer.clear()
        self.prev_gray = None
        self.prediction_label.setText("Prediction: ")

    def update_frame(self):
        """Capture and process each frame."""
        ret, frame = self.cap.read()
        if not ret:
            return

        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(frame_rgb)

        # Process hands
        skeletal_data = []
        crops_data = []
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Skeletal data: 21 landmarks with (x, y, z)
                landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                skeletal_data.append(landmarks)

                # Hand crop using your utilities
                x_min, y_min, x_max, y_max = get_bounding_box(hand_landmarks, self.frame_width, self.frame_height)
                crop = frame[y_min:y_max, x_min:x_max]
                crop_resized = resize_preserve_aspect_ratio(crop, (112, 112))
                crop_resized = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB) / 255.0  # Normalize to [0, 1]
                crops_data.append(crop_resized)

                # Draw landmarks and bounding box on frame
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Pad to 2 hands if fewer detected
        while len(skeletal_data) < 2:
            skeletal_data.append([[0.0, 0.0, 0.0]] * 21)
        while len(crops_data) < 2:
            crops_data.append(np.zeros((112, 112, 3)))

        # Optical flow computation
        optical_flow = np.zeros((2, 112, 112))  # Default zero flow
        if self.prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(self.prev_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_resized = cv2.resize(flow, (112, 112))
            optical_flow = flow_resized.transpose(2, 0, 1)  # Shape: (2, 112, 112)
        self.prev_gray = frame_gray

        # Append to buffers
        self.skeletal_buffer.append(skeletal_data)
        self.crops_buffer.append(crops_data)
        self.optical_flow_buffer.append([optical_flow, optical_flow])  # Duplicate for 2 hands

        # Maintain buffer size
        if len(self.skeletal_buffer) > self.sequence_length:
            self.skeletal_buffer.pop(0)
            self.crops_buffer.pop(0)
            self.optical_flow_buffer.pop(0)

        # Predict when buffer is full
        if len(self.skeletal_buffer) == self.sequence_length:
            # Prepare tensors for dataloader/model
            skeletal_tensor = torch.tensor(self.skeletal_buffer, dtype=torch.float32).to(self.device)  # (T, 2, 21, 3)
            crops_tensor = torch.tensor(self.crops_buffer, dtype=torch.float32).permute(0, 2, 3, 1).to(self.device)  # (T, 2, 112, 112, 3)
            optical_flow_tensor = torch.tensor(self.optical_flow_buffer, dtype=torch.float32).to(self.device)  # (T, 2, 2, 112, 112)
            input_lengths = torch.tensor([self.sequence_length], dtype=torch.long).to(self.device)

            # Add batch dimension
            skeletal_tensor = skeletal_tensor.unsqueeze(0)  # (1, T, 2, 21, 3)
            crops_tensor = crops_tensor.unsqueeze(0)  # (1, T, 2, 112, 112, 3)
            optical_flow_tensor = optical_flow_tensor.unsqueeze(0)  # (1, T, 2, 2, 112, 112)

            # Model inference
            with torch.no_grad():
                pred_sequences = self.model.decode(skeletal_tensor, crops_tensor, optical_flow_tensor, input_lengths)
                pred_glosses = ' '.join([self.idx_to_gloss[idx] for idx in pred_sequences[0]])
                self.prediction_label.setText(f"Prediction: {pred_glosses}")

        # Display the frame
        frame_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_display.shape
        bytes_per_line = ch * w
        q_image = QImage(frame_display.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

    def closeEvent(self, event):
        """Clean up resources on window close."""
        self.cap.release()
        self.mp_hands.close()
        self.timer.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CSLRWindow()
    window.show()
    sys.exit(app.exec_())