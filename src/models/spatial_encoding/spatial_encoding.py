# models/spatial_encoding/spatial_encoding.py
import torch
import torch.nn as nn
from torchvision import models

class Graph:
    def __init__(self, num_nodes=21, device='cpu'):
        self.num_nodes = num_nodes
        self.device = torch.device(device)
        self.A = self.build_hand_adjacency()

    def build_hand_adjacency(self):
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),      # Index
            (0, 9), (9, 10), (10, 11), (11, 12), # Middle
            (0, 13), (13, 14), (14, 15), (15, 16), # Ring
            (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
        ]
        A = torch.zeros(self.num_nodes, self.num_nodes, device=self.device)
        for i, j in connections:
            A[i, j] = 1
            A[j, i] = 1
        A = A + torch.eye(self.num_nodes, device=self.device)
        D = torch.diag(A.sum(1) ** -0.5)
        A = D @ A @ D
        return A

class GraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, A, device='cpu'):
        super(GraphConvLayer, self).__init__()
        self.device = torch.device(device)
        self.gcn = nn.Linear(in_channels, out_channels)
        self.A = A.to(self.device)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = x.to(self.device)
        N, V, C = x.shape
        x = x.permute(0, 2, 1)  # (N, V, C) -> (N, C, V)
        x = torch.einsum("vu, ncu -> ncv", self.A, x)  # Graph convolution
        x = x.reshape(-1, C)  # (N*V, C)
        x = self.gcn(x)  # (N*V, out_channels)
        x = x.view(N, V, -1)  # (N, V, out_channels)
        x = self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = nn.functional.relu(x)
        return x

class SpatialEncoding(nn.Module):
    def __init__(self, D_spatial=128, device='cpu'):
        super(SpatialEncoding, self).__init__()
        self.device = torch.device(device)
        
        # Graph for skeletal data
        self.graph = Graph(num_nodes=21, device=device)
        A = self.graph.A
        
        # Graph convolution layers (per frame)
        self.graph_layer1 = GraphConvLayer(3, 32, A, device)  # 3 -> 32
        self.graph_layer2 = GraphConvLayer(32, 32, A, device)  # 32 -> 32
        
        # CNNs for RGB and Optical Flow (default width_mult=1.0)
        self.cnn_rgb = models.mobilenet_v3_small(weights='DEFAULT')
        self.cnn_flow = models.mobilenet_v3_small(weights='DEFAULT')
        self.cnn_flow.features[0][0] = nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Define CNN blocks (default channels: 16 and 24)
        self.first_block_rgb = nn.Sequential(*self.cnn_rgb.features[:2])  # Outputs 16 channels
        self.second_block_rgb = nn.Sequential(*self.cnn_rgb.features[2:4])  # Outputs 24 channels
        self.first_block_flow = nn.Sequential(*self.cnn_flow.features[:2])  # Outputs 16 channels
        self.second_block_flow = nn.Sequential(*self.cnn_flow.features[2:4])  # Outputs 24 channels
        
        # MLPs for interleaved fusion
        self.mlp1 = nn.Sequential(  # First fusion: 32 (skeletal) + 16 (RGB) + 16 (flow)
            nn.Linear(32 + 16 + 16, 64),
            nn.ReLU(),
            nn.Linear(64, 32 + 16 + 16)
        )
        self.mlp2 = nn.Sequential(  # Second fusion: 32 (skeletal) + 24 (RGB) + 24 (flow)
            nn.Linear(32 + 24 + 24, 64),
            nn.ReLU(),
            nn.Linear(64, 32 + 24 + 24)
        )
        self.final_mlp = nn.Sequential(  # Final fusion: 32 + 24 + 24 -> 128
            nn.Linear(32 + 24 + 24, 128),
            nn.ReLU(),
            nn.Linear(128, D_spatial)
        )
        
        # Normalization for RGB
        self.rgb_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.rgb_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        
        self.to(self.device)

    def forward(self, skeletal, crops, optical_flow):
        B, T, _, _, _ = skeletal.shape
        
        # Flatten inputs
        skeletal_flat = skeletal.reshape(B*T*2, 21, 3).to(self.device)
        crops_flat = crops.reshape(B*T*2, 3, 112, 112).to(self.device)
        flow_flat = optical_flow.reshape(B*T*2, 2, 112, 112).to(self.device)
        
        # Normalize RGB inputs
        crops_flat = (crops_flat - self.rgb_mean) / self.rgb_std
        
        # First block
        skeletal_feat1 = self.graph_layer1(skeletal_flat)  # (B*T*2, 21, 32)
        rgb_feat1 = self.first_block_rgb(crops_flat)  # (B*T*2, 16, H1, W1)
        flow_feat1 = self.first_block_flow(flow_flat)  # (B*T*2, 16, H2, W2)
        
        # Pool for first fusion
        skeletal_pooled1 = skeletal_feat1.mean(dim=1)  # (B*T*2, 32)
        rgb_pooled1 = rgb_feat1.mean(dim=(2, 3))  # (B*T*2, 16)
        flow_pooled1 = flow_feat1.mean(dim=(2, 3))  # (B*T*2, 16)
        
        concat1 = torch.cat([skeletal_pooled1, rgb_pooled1, flow_pooled1], dim=1)  # (B*T*2, 64)
        fused1 = self.mlp1(concat1)  # (B*T*2, 64)
        skeletal_fused1, rgb_fused1, flow_fused1 = torch.split(fused1, [32, 16, 16], dim=1)
        
        # Add residuals back
        skeletal_feat1 = skeletal_feat1 + skeletal_fused1.unsqueeze(1)
        rgb_feat1 = rgb_feat1 + rgb_fused1.unsqueeze(2).unsqueeze(3)
        flow_feat1 = flow_feat1 + flow_fused1.unsqueeze(2).unsqueeze(3)
        
        # Second block
        skeletal_feat2 = self.graph_layer2(skeletal_feat1)  # (B*T*2, 21, 32)
        rgb_feat2 = self.second_block_rgb(rgb_feat1)  # (B*T*2, 24, H3, W3)
        flow_feat2 = self.second_block_flow(flow_feat1)  # (B*T*2, 24, H4, W4)
        
        # Pool for second fusion
        skeletal_final = skeletal_feat2.mean(dim=1)  # (B*T*2, 32)
        rgb_final = rgb_feat2.mean(dim=(2, 3))  # (B*T*2, 24)
        flow_final = flow_feat2.mean(dim=(2, 3))  # (B*T*2, 24)
        
        concat2 = torch.cat([skeletal_final, rgb_final, flow_final], dim=1)  # (B*T*2, 80)
        fused2 = self.mlp2(concat2)  # (B*T*2, 80)
        skeletal_fused2, rgb_fused2, flow_fused2 = torch.split(fused2, [32, 24, 24], dim=1)
        
        # Add residuals (optional, included for consistency)
        skeletal_final = skeletal_final + skeletal_fused2
        rgb_final = rgb_final + rgb_fused2
        flow_final = flow_final + flow_fused2
        
        # Final fusion
        final_concat = torch.cat([skeletal_final, rgb_final, flow_final], dim=1)  # (B*T*2, 80)
        spatial_features = self.final_mlp(final_concat)  # (B*T*2, D_spatial)
        
        # Reshape to (B, T, 2, D_spatial)
        spatial_features = spatial_features.view(B, T, 2, -1)
        return spatial_features

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SpatialEncoding(D_spatial=128, device=device)
    skeletal = torch.randn(4, 114, 2, 21, 3)
    crops = torch.randn(4, 114, 2, 3, 112, 112)
    optical_flow = torch.randn(4, 114, 2, 2, 112, 112)
    output = model(skeletal, crops, optical_flow)
    print(f"Output shape: {output.shape}")  # Expected: (4, 114, 2, 128)