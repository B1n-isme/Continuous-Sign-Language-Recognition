# models/spatial_encoding/spatial_encoding.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.spatial_encoding.gcn import STGCN
from models.spatial_encoding.cnn import CNNEncoder


class SpatialEncoding(nn.Module):
    """
    Spatial encoding module that processes skeletal data, hand crops, and optical flow.

    Args:
        D_skeletal: Dimension of skeletal features
        D_cnn: Dimension of CNN features for crops
        D_flow: Dimension of optical flow features
    """

    def __init__(self, D_skeletal=64, D_cnn=512, D_flow=512):
        super(SpatialEncoding, self).__init__()
        # Skeletal encoding for each hand (could share weights if desired)
        self.stgcn_left = STGCN(in_channels=3, out_dim=D_skeletal)
        self.stgcn_right = STGCN(in_channels=3, out_dim=D_skeletal)
        # CNN for crops (RGB, 3 channels)
        self.cnn_crops = CNNEncoder(in_channels=3, output_dim=D_cnn)
        # CNN for optical flow (2 channels)
        self.cnn_flow = CNNEncoder(in_channels=2, output_dim=D_flow)

    def forward(self, skeletal, crops, optical_flow):
        """
        Forward pass through the spatial encoding module.

        Args:
            skeletal: Tensor of shape (B, T, 2, 21, 3) - batch, time, hands, joints, coords
            crops: Tensor of shape (B, T, 2, 3, 112, 112) - batch, time, hands, channels, height, width
            optical_flow: Tensor of shape (B, T-1, 2, 2, 112, 112) - batch, time, hands, flow_channels, height, width

        Returns:
            Tensor of shape (B, T, 2, D_skeletal + D_cnn + D_flow) containing combined features
        """
        B, T, _, _, _ = skeletal.shape

        # 1. Skeletal Encoding
        skeletal_left = self.stgcn_left(skeletal[:, :, 0])  # (B, T, 64)
        skeletal_right = self.stgcn_right(skeletal[:, :, 1])  # (B, T, 64)
        skeletal_features = torch.stack(
            [skeletal_left, skeletal_right], dim=2
        )  # (B, T, 2, 64)

        # 2. Crops Encoding
        crops_flat = crops.reshape(B * T * 2, 3, 112, 112)  # (B*T*2, 3, 112, 112)
        crops_features = self.cnn_crops(crops_flat)  # (B*T*2, 512)
        crops_features = crops_features.view(B, T, 2, -1)  # (B, T, 2, 512)

        # 3. Optical Flow Encoding (pad to T=191)
        # Create zero tensor for the last time step that's missing
        zero_pad = torch.zeros_like(optical_flow[:, :1, :, :, :, :])
        optical_flow_padded = torch.cat(
            [optical_flow, zero_pad], dim=1
        )  # (B, T, 2, 2, 112, 112)

        # Correctly reshape the tensor for processing
        flow_flat = optical_flow_padded.reshape(
            B * T * 2, 2, 112, 112
        )  # (B*T*2, 2, 112, 112)
        flow_features = self.cnn_flow(flow_flat)  # (B*T*2, 512)
        flow_features = flow_features.view(B, T, 2, -1)  # (B, T, 2, 512)

        # 4. Concatenate Features per Hand
        hand_features = torch.cat(
            [skeletal_features, crops_features, flow_features], dim=3
        )  # (B, T, 2, 1088)

        return hand_features  # Shape: (B, T, 2, 64 + 512 + 512)


# Example usage (for testing)
if __name__ == "__main__":
    # Sample input: batch_size=4, sequence_length=191, num_hands=2
    skeletal = torch.randn(4, 191, 2, 21, 3)
    crops = torch.randn(4, 191, 2, 3, 112, 112)
    optical_flow = torch.randn(4, 190, 2, 2, 112, 112)
    model = SpatialEncoding(D_skeletal=64, D_cnn=512, D_flow=512)
    output = model(skeletal, crops, optical_flow)
    print(output.shape)
