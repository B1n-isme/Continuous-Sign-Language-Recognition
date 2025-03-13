# models/spatial_encoding/spatial_encoding.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.spatial_encoding.cnn import CNNEncoder  # Adjust path as needed
from src.models.spatial_encoding.gcn import STGCN       # Adjust path as needed

class SpatialEncoding(nn.Module):
    """
    Spatial encoding module that processes skeletal data, hand crops, and optical flow.

    Args:
        D_skeletal: Dimension of skeletal features (per hand)
        D_cnn: Dimension of CNN features for crops
        D_flow: Dimension of optical flow features
    """
    def __init__(self, D_skeletal=64, D_cnn=512, D_flow=512):
        super(SpatialEncoding, self).__init__()
        
        # Skeletal encoding: STGCN for both hands (2D keypoints)
        self.stgcn = STGCN(in_channels=2, out_dim=D_skeletal, num_joints=21, num_layers=2)
        
        # CNN for crops (RGB, 3 channels)
        self.cnn_crops = CNNEncoder(in_channels=3, output_dim=D_cnn, modality="rgb")
        
        # CNN for optical flow (2 channels)
        self.cnn_flow = CNNEncoder(in_channels=2, output_dim=D_flow, modality="flow")

    def forward(self, skeletal, crops, optical_flow):
        """
        Forward pass through the spatial encoding module.

        Args:
            skeletal: Tensor of shape (B, T, 2, 21, 3) - batch, time, hands, joints, coords (x, y, z)
            crops: Tensor of shape (B, T, 2, 3, 112, 112) - batch, time, hands, channels, height, width
            optical_flow: Tensor of shape (B, T-1, 2, 2, 112, 112) - batch, time, hands, flow_channels, height, width

        Returns:
            Tensor of shape (B, T, 2, D_skeletal + D_cnn + D_flow) containing combined features
        """
        B, T, _, _, _ = skeletal.shape
        T_flow = optical_flow.shape[1]  # T-1

        # 1. Skeletal Encoding
        skeletal_2d = skeletal[..., :2]  # (B, T, 2, 21, 2)
        skeletal_features = self.stgcn(skeletal_2d)  # (B, T, 128)
        skeletal_features = skeletal_features.view(B, T, 2, -1)  # (B, T, 2, 64)

        # 2. Crops Encoding
        crops_flat = crops.reshape(B * T * 2, 3, 112, 112)  # (B*T*2, 3, 112, 112)
        crops_features = self.cnn_crops(crops_flat)  # (B*T*2, 512)
        crops_features = crops_features.view(B, T, 2, -1)  # (B, T, 2, 512)

        # 3. Optical Flow Encoding
        optical_flow_flat = optical_flow.reshape(B * T_flow * 2, 2, 112, 112)  # (B*(T-1)*2, 2, 112, 112)
        flow_features_flat = self.cnn_flow(optical_flow_flat)  # (B*(T-1)*2, 512)
        flow_features = flow_features_flat.view(B, T_flow, 2, -1)  # (B, T-1, 2, 512)
        
        # Interpolate flow features to match T
        flow_features_reshaped = flow_features.reshape(B * 2, 512, T_flow)  # (B*2, 512, T-1)
        flow_features_interp = F.interpolate(
            flow_features_reshaped,  # (B*2, 512, T-1)
            size=T, mode='linear', align_corners=False
        )  # (B*2, 512, T)
        flow_features = flow_features_interp.view(B, T, 2, 512)  # (B, T, 2, 512)

        # 4. Concatenate Features per Hand
        hand_features = torch.cat(
            [skeletal_features, crops_features, flow_features], dim=3
        )  # (B, T, 2, 64 + 512 + 512) = (B, T, 2, 1088)

        return hand_features

# Example usage (for testing)
if __name__ == "__main__":
    # Sample input: batch_size=4, sequence_length=104
    skeletal = torch.randn(4, 191, 2, 21, 3)
    crops = torch.randn(4, 191, 2, 3, 112, 112)
    optical_flow = torch.randn(4, 190, 2, 2, 112, 112)
    model = SpatialEncoding(D_skeletal=64, D_cnn=512, D_flow=512)
    output = model(skeletal, crops, optical_flow)
    print(f"Output shape: {output.shape}")  # Expected: (4, 104, 2, 1088)