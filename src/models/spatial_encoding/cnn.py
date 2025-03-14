# models/spatial_encoding/cnn.py
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import functional as F

class SqueezeExcitation(nn.Module):
    def __init__(self, channel, reduction=4, device='cpu'):
        super(SqueezeExcitation, self).__init__()
        self.fc1 = nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        scale = nn.functional.adaptive_avg_pool2d(x, 1)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.sigmoid(scale)
        return x * scale

class CNNEncoder(nn.Module):
    def __init__(self, in_channels=3, output_dim=512, modality="rgb", device='cpu'):
        super(CNNEncoder, self).__init__()
        self.device = torch.device(device)
        self.in_channels = in_channels
        self.modality = modality.lower()

        # MobileNetV3 Small backbone
        self.cnn = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        if in_channels != 3:
            self.cnn.features[0][0] = nn.Conv2d(
                in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False
            )
        self.cnn.classifier = nn.Identity()

        # Optimized head
        self.head = nn.Sequential(
            nn.Conv2d(576, 256, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=256),
            nn.ReLU(inplace=True),
            SqueezeExcitation(256, reduction=4),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(256, output_dim)

        # Normalization parameters
        if self.modality == "rgb":
            self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        else:  # Optical flow
            self.mean = torch.zeros(in_channels, device=self.device).view(1, in_channels, 1, 1)
            self.std = torch.ones(in_channels, device=self.device).view(1, in_channels, 1, 1)

        # Freeze early layers
        for param in self.cnn.features[:5].parameters():
            param.requires_grad = False

        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)  # (B*T*2, C, H, W)
        x = F.normalize(x, mean=self.mean, std=self.std)
        features = self.cnn.features(x)  # (B*T*2, 576, H', W')
        features = self.head(features)   # (B*T*2, 256, 1, 1)
        features = features.view(features.size(0), -1)  # (B*T*2, 256)
        features = self.fc(features)     # (B*T*2, output_dim)
        return features

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cnn_rgb = CNNEncoder(in_channels=3, output_dim=512, modality="rgb", device=device)
    cnn_flow = CNNEncoder(in_channels=2, output_dim=512, modality="flow", device=device)
    dummy_input_rgb = torch.randn(8, 3, 112, 112).to(device)
    dummy_input_flow = torch.randn(8, 2, 112, 112).to(device)
    print("RGB output shape:", cnn_rgb(dummy_input_rgb).shape)  # (8, 512)
    print("Flow output shape:", cnn_flow(dummy_input_flow).shape)  # (8, 512)