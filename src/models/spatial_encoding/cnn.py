import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import functional as F

class SqueezeExcitation(nn.Module):
    def __init__(self, channel, reduction=4, device='cpu'):
        super(SqueezeExcitation, self).__init__()
        self.device = torch.device(device)
        self.fc1 = nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=True).to(self.device)
        self.relu = nn.ReLU(inplace=True).to(self.device)
        self.fc2 = nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=True).to(self.device)
        self.sigmoid = nn.Sigmoid().to(self.device)

    def forward(self, x):
        # Squeeze: global spatial pooling
        scale = nn.functional.adaptive_avg_pool2d(x, 1)
        # Excitation: learn channel-wise dependencies
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
        self.modality = modality.lower()  # "rgb" or "flow"

        # Shared backbone: MobileNetV3 Small feature extractor
        self.cnn = models.mobilenet_v3_small(weights=None)
        state_dict = torch.load("D:/Data/mobilenet_v3_small-047dcff4.pth", map_location=self.device)
        self.cnn.load_state_dict(state_dict)
        # self.cnn = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.cnn.eval()

        if in_channels != 3:
            # Adjust first conv layer for optical flow (or different modalities)
            self.cnn.features[0][0] = nn.Conv2d(
                in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False
            ).to(self.device)
        
        # Remove classification head
        self.cnn.classifier = nn.Identity()

        # Optimized head with 1x1 conv, GroupNorm, ReLU, SE block and adaptive pooling
        self.head = nn.Sequential(
            nn.Conv2d(576, 256, kernel_size=1, bias=False),   # Reduce channels
            nn.GroupNorm(num_groups=8, num_channels=256),       # More robust than BatchNorm for small batches
            nn.ReLU(inplace=True),
            SqueezeExcitation(256, reduction=4),                # Attention mechanism
            nn.AdaptiveAvgPool2d(1)                             # Global pooling to (B*T*2, 256, 1, 1)
        ).to(self.device)
        self.fc = nn.Linear(256, output_dim).to(self.device)  # Final feature dimension

        # Normalization parameters
        if self.modality == "rgb":
            self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        else:  # Optical flow: Assuming preprocessed to [-1, 1]
            self.mean = torch.zeros(in_channels, device=self.device).view(1, in_channels, 1, 1)
            self.std = torch.ones(in_channels, device=self.device).view(1, in_channels, 1, 1)

        # Freeze early layers to reduce training time and computation
        for param in self.cnn.features[:5].parameters():
            param.requires_grad = False

        # Move model to device
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        # x: (B*T*2, C, H, W), e.g., (B*T*2, 3, 112, 112) or (B*T*2, 2, 112, 112)
        # Normalize input
        x = F.normalize(x, mean=self.mean, std=self.std)
        
        # Shared feature extraction
        features = self.cnn.features(x)  # (B*T*2, 576, H', W')
        
        # Optimized head processing
        features = self.head(features)   # (B*T*2, 256, 1, 1)
        features = features.view(features.size(0), -1)  # Flatten to (B*T*2, 256)
        features = self.fc(features)     # (B*T*2, output_dim)
        return features

# Usage example
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize models with device
    cnn_rgb = CNNEncoder(in_channels=3, output_dim=512, modality="rgb", device=device)
    cnn_flow = CNNEncoder(in_channels=2, output_dim=512, modality="flow", device=device)
    
    # Example forward pass with dummy data
    dummy_input_rgb = torch.randn(8, 3, 112, 112).to(device)
    dummy_input_flow = torch.randn(8, 2, 112, 112).to(device)

    print("RGB output shape:", cnn_rgb(dummy_input_rgb).shape)  # Expected: (8, 512)
    print("Flow output shape:", cnn_flow(dummy_input_flow).shape)  # Expected: (8, 512)
