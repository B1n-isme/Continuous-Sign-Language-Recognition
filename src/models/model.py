import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
from torch_geometric.nn import STConv

class CSLModel(nn.Module):
    def __init__(self, num_glosses):
        super().__init__()
        # RGB + Flow CNN (MobileNetV3-Small)
        self.rgb_cnn = mobilenet_v3_small(pretrained=True)
        self.rgb_cnn.features[0][0] = nn.Conv2d(5, 16, 3, stride=2, padding=1)  # 3 RGB + 2 Flow
        self.rgb_dim = 576  # MobileNetV3-Small output

        # Skeletal ST-GCN
        self.st_gcn = STConv(3, 64, kernel_size=(3, 1), stride=1)  # Input: (N, C=3, T, V=21)

        # Temporal Encoding (TempConv)
        self.temp_conv = nn.Conv1d(self.rgb_dim + 64, 256, kernel_size=5, padding=2)

        # GRU for Long-term Dependencies
        self.gru = nn.GRU(256, 256, num_layers=1, batch_first=True)

        # Transformer for Sequence Learning
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=4), num_layers=2)

        # Output Layer (Gloss Prediction)
        self.fc = nn.Linear(256, num_glosses)

    def forward(self, rgb, flow, skeletal):
        B, T, H = rgb.shape[0], rgb.shape[1], rgb.shape[2]  # Batch, Time, Hands
        
        # RGB + Flow: (B, T*H, 5, 112, 112)
        rgb_flow = torch.cat([rgb, flow], dim=-3)  # Concatenate along channel dim
        rgb_flow = rgb_flow.view(B * T * H, 5, 112, 112)
        rgb_features = self.rgb_cnn.features(rgb_flow)  # (B*T*H, 576, 7, 7)
        rgb_features = rgb_features.mean([2, 3])  # Global avg pool: (B*T*H, 576)
        rgb_features = rgb_features.view(B, T * H, -1)  # (B, T*H, 576)

        # Skeletal: (B, 3, T, 21*H)
        skeletal = skeletal.view(B, 3, T, -1)
        skeletal_features = self.st_gcn(skeletal)  # (B, 64, T, 21*H)
        skeletal_features = skeletal_features.mean(-1)  # (B, 64, T)

        # Fusion: (B, T, 576+64)
        fused = torch.cat([rgb_features.view(B, T, H, -1).mean(2), skeletal_features.transpose(1, 2)], dim=-1)
        
        # TempConv: (B, 256, T)
        fused = self.temp_conv(fused.transpose(1, 2))
        
        # GRU: (B, T, 256)
        gru_out, _ = self.gru(fused.transpose(1, 2))
        
        # Transformer: (T, B, 256)
        transformer_out = self.transformer(gru_out.transpose(0, 1))
        
        # Output: (B, T, num_glosses)
        return self.fc(transformer_out.transpose(0, 1))

# Instantiate (assume 50 glosses in vocab)
model = CSLModel(num_glosses=50)