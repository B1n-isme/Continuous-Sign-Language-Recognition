import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import functional as F

class CNNEncoder(nn.Module):
    def __init__(self, in_channels=3, output_dim=512, modality="rgb"):
        super(CNNEncoder, self).__init__()
        self.in_channels = in_channels
        self.modality = modality.lower()  # "rgb" or "flow"

        # Shared backbone: MobileNetV3 Small feature extractor
        # self.cnn = models.mobilenet_v3_small(weights=None)
        # state_dict = torch.load("D:/Data/mobilenet_v3_small-047dcff4.pth")
        # self.cnn.load_state_dict(state_dict)
        self.cnn = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.cnn.eval()


        if in_channels != 3:
            # Adjust first conv layer for optical flow (2 channels)
            self.cnn.features[0][0] = nn.Conv2d(
                in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False
            )
        
        # Remove classification head
        self.cnn.classifier = nn.Identity()

        # Modality-specific heads
        self.head = nn.Sequential(
            nn.Conv2d(576, 256, kernel_size=1),  # Reduce channels
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # Global pooling to (B*T*2, 256, 1, 1)
        )
        self.fc = nn.Linear(256, output_dim)  # Final feature dimension

        # Normalization parameters
        if self.modality == "rgb":
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
        else:  # Optical flow: Assuming preprocessed to [-1, 1]
            self.mean = [0.0] * in_channels  # Adjust based on your preprocessing
            self.std = [1.0] * in_channels

        # Optional: Freeze early layers
        for param in self.cnn.features[:5].parameters():
            param.requires_grad = False

    def forward(self, x):
        # x: (B*T*2, C, H, W), e.g., (B*T*2, 3, 112, 112) or (B*T*2, 2, 112, 112)
        # Normalize input
        x = F.normalize(x, mean=self.mean, std=self.std)
        
        # Shared feature extraction
        features = self.cnn.features(x)  # (B*T*2, 576, H', W')
        
        # Modality-specific head
        features = self.head(features)  # (B*T*2, 256, 1, 1)
        features = features.squeeze(-1).squeeze(-1)  # (B*T*2, 256)
        features = self.fc(features)  # (B*T*2, 512)
        return features

# Usage example
if __name__ == "__main__":
    # For cropped images
    cnn_rgb = CNNEncoder(in_channels=3, output_dim=512, modality="rgb")
    # For optical flow
    cnn_flow = CNNEncoder(in_channels=2, output_dim=512, modality="flow")