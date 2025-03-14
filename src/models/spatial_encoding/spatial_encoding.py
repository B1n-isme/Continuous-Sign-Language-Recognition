# models/spatial_encoding/spatial_encoding.py
import torch
import torch.nn as nn

from src.models.spatial_encoding.cnn import CNNEncoder
from src.models.spatial_encoding.gcn import STGCN

class SpatialEncoding(nn.Module):
    def __init__(self, D_skeletal=64, D_cnn=512, D_flow=512, device='cpu'):
        super(SpatialEncoding, self).__init__()
        self.device = torch.device(device)
        
        # Skeletal encoding with 3D keypoints
        self.stgcn = STGCN(in_channels=3, out_dim=D_skeletal, num_joints=21, num_layers=2, device=device)
        self.cnn_crops = CNNEncoder(in_channels=3, output_dim=D_cnn, modality="rgb", device=device)
        self.cnn_flow = CNNEncoder(in_channels=2, output_dim=D_flow, modality="flow", device=device)

    def forward(self, skeletal, crops, optical_flow):
        """
        Args:
            skeletal: (B, T, 2, 21, 3) - 3D keypoints
            crops: (B, T, 2, 3, 112, 112) - RGB hand crops
            optical_flow: (B, T, 2, 2, 112, 112) - Padded optical flow

        Returns:
            (B, T, 2, D_skeletal + D_cnn + D_flow) - Combined features
        """
        skeletal, crops, optical_flow = skeletal.to(self.device), crops.to(self.device), optical_flow.to(self.device)
        B, T, _, _, _ = skeletal.shape

        # 1. Skeletal Encoding (3D)
        skeletal_features = self.stgcn(skeletal)  # (B, T, 2, 64)

        # 2. Crops Encoding
        crops_flat = crops.reshape(B * T * 2, 3, 112, 112)  # (B*T*2, 3, 112, 112)
        crops_features = self.cnn_crops(crops_flat)  # (B*T*2, 512)
        crops_features = crops_features.view(B, T, 2, -1)  # (B, T, 2, 512)

        # 3. Optical Flow Encoding (no interpolation)
        optical_flow_flat = optical_flow.reshape(B * T * 2, 2, 112, 112)  # (B*T*2, 2, 112, 112)
        flow_features = self.cnn_flow(optical_flow_flat)  # (B*T*2, 512)
        flow_features = flow_features.view(B, T, 2, -1)  # (B, T, 2, 512)

        # 4. Concatenate Features
        hand_features = torch.cat([skeletal_features, crops_features, flow_features], dim=3)  # (B, T, 2, 1088)
        return hand_features

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    skeletal = torch.randn(4, 191, 2, 21, 3).to(device)
    crops = torch.randn(4, 191, 2, 3, 112, 112).to(device)
    optical_flow = torch.randn(4, 191, 2, 2, 112, 112).to(device)  # Updated to T=191
    model = SpatialEncoding(D_skeletal=64, D_cnn=512, D_flow=512, device=device)
    output = model(skeletal, crops, optical_flow)
    print(f"Output shape: {output.shape}")  # (4, 191, 2, 1088)