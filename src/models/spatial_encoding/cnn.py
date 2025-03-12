import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V3_Small_Weights


class CNNEncoder(nn.Module):
    def __init__(self, in_channels=3, output_dim=512):
        super(CNNEncoder, self).__init__()
        # Fix: Replace pretrained=True with weights=DEFAULT
        self.cnn = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        if in_channels != 3:
            # Adjust input channels (e.g., for optical flow with 2 channels)
            self.cnn.features[0][0] = nn.Conv2d(
                in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False
            )
        self.cnn.classifier = nn.Identity()  # Remove classification head
        self.fc = nn.Linear(576, output_dim)  # MobileNetV3 outputs 576 features

    def forward(self, x):
        # x: (B*T*2, 3, 112, 112)
        features = self.cnn(x)  # (B*T*2, 576)
        features = self.fc(features)  # (B*T*2, 512)
        return features
