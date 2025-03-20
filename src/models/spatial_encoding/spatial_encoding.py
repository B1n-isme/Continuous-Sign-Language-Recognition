# models/spatial_encoding/spatial_encoding.py
import torch
import torch.nn as nn
import torch.nn.functional as F
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

# ---------------------------
# SE Module Definition
# ---------------------------
class SEModule(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEModule, self).__init__()
        self.fc1 = nn.Conv1d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv1d(channels // reduction, channels, kernel_size=1)
    
    def forward(self, x):
        # x shape: (B, channels, 1) assuming we squeeze spatially
        se_weight = F.adaptive_avg_pool1d(x, 1)  # (B, channels, 1)
        se_weight = F.relu(self.fc1(se_weight))
        se_weight = torch.sigmoid(self.fc2(se_weight))
        return x * se_weight

# ---------------------------
# Depthwise Separable MLP with SE for Fusion
# ---------------------------
class DepthwiseSeparableMLPSE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(DepthwiseSeparableMLPSE, self).__init__()
        # Depthwise convolution: groups = in_channels (simulate channel-wise transformation)
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=1, groups=in_channels)
        # Pointwise convolution: combine channels linearly
        self.pointwise = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.activation = nn.ReLU(inplace=True)
        self.se = SEModule(hidden_channels, reduction=2)
        self.fc = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x):
        # x: (B, in_channels)
        # Reshape to (B, in_channels, 1)
        x = x.unsqueeze(-1)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.activation(x)
        x = self.se(x)
        # Squeeze the last dimension: (B, hidden_channels)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x

# ---------------------------
# Modified Cross-Modal Fusion Block with Multimodal MLP Fusion
# ---------------------------
class CrossModalFusionBlockSE(nn.Module):
    """
    This block projects each modality’s pooled features to a common space,
    uses self-attention (via nn.MultiheadAttention) to allow them to interact,
    and then applies a learned gating mechanism with a depthwise separable MLP
    combined with an SE module for cross-modal awareness.
    """
    def __init__(self, dims, common_dim):
        """
        dims: list of input dimensions for each modality [dim1, dim2, ...]
        common_dim: common embedding dimension for attention (e.g., 16 or 24)
        """
        super().__init__()
        self.num_modalities = len(dims)
        # Project each modality to common_dim
        self.projs = nn.ModuleList([nn.Linear(d, common_dim) for d in dims])
        # Reverse projection if needed (if original dim != common_dim)
        self.rev_projs = nn.ModuleList([
            nn.Identity() if d == common_dim else nn.Linear(common_dim, d) for d in dims
        ])
        # Multihead attention with a single head
        self.attention = nn.MultiheadAttention(embed_dim=common_dim, num_heads=1, batch_first=True)
        # Replace the simple gating MLP with our depthwise separable MLP+SE block
        self.gates = nn.ModuleList([
            DepthwiseSeparableMLPSE(in_channels=common_dim * 2, hidden_channels=common_dim, out_channels=common_dim)
            for _ in range(self.num_modalities)
        ])

    def forward(self, feats):
        """
        feats: list of modality features, each of shape (B, d_i)
        Returns:
            updated: list of updated modality features, same shapes as input.
        """
        B = feats[0].size(0)
        # Project each modality and add token dimension: (B, 1, common_dim)
        tokens = [proj(f).unsqueeze(1) for f, proj in zip(feats, self.projs)]
        tokens = torch.cat(tokens, dim=1)  # (B, num_modalities, common_dim)
        # Self-attention across modalities
        attn_output, _ = self.attention(tokens, tokens, tokens)
        updated = []
        # For each modality, compute a gating value with the new MLP+SE block and update
        for i in range(self.num_modalities):
            orig = tokens[:, i, :]       # (B, common_dim)
            attn_i = attn_output[:, i, :]  # (B, common_dim)
            # Concatenate original and attended features
            gate_input = torch.cat([orig, attn_i], dim=1)  # (B, common_dim*2)
            # Compute gate using depthwise separable MLP with SE
            gate = self.gates[i](gate_input)  # (B, common_dim)
            new = orig + gate * attn_i
            # Reverse projection if needed
            new = self.rev_projs[i](new)
            updated.append(new)
        return updated

# ---------------------------
# Updated SpatialEncoding Module with SE-based Fusion
# ---------------------------
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
        
        # CNNs for RGB and Optical Flow (using MobileNet V3 Small)
        self.cnn_rgb = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        # self.cnn_rgb = models.mobilenet_v3_small(weights=None)
        # state_dict = torch.load("D:/Data/mobilenet_v3_small-047dcff4.pth", map_location=self.device)
        # self.cnn_rgb.load_state_dict(state_dict)
        self.cnn_rgb.eval()

        self.cnn_flow = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        # self.cnn_flow = models.mobilenet_v3_small(weights=None)
        # state_dict = torch.load("D:/Data/mobilenet_v3_small-047dcff4.pth", map_location=self.device)
        # self.cnn_flow.load_state_dict(state_dict)
        self.cnn_flow.eval()


        # Adjust first conv layer for optical flow (2 channels)
        self.cnn_flow.features[0][0] = nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Define CNN blocks: first and second blocks for each modality
        self.first_block_rgb = nn.Sequential(*self.cnn_rgb.features[:2])  # Outputs 16 channels
        self.second_block_rgb = nn.Sequential(*self.cnn_rgb.features[2:4])  # Outputs 24 channels
        self.first_block_flow = nn.Sequential(*self.cnn_flow.features[:2])  # Outputs 16 channels
        self.second_block_flow = nn.Sequential(*self.cnn_flow.features[2:4])  # Outputs 24 channels
        
        # Replace fusion blocks with our new SE-based cross-modal fusion blocks:
        # First fusion: skeletal (32), rgb (16), flow (16) -> common dim = 16
        self.fusion_block1 = CrossModalFusionBlockSE(dims=[32, 16, 16], common_dim=16)
        # Second fusion: skeletal (32), rgb (24), flow (24) -> common dim = 24
        self.fusion_block2 = CrossModalFusionBlockSE(dims=[32, 24, 24], common_dim=24)
        
        # Final fusion: maps concatenated features (32+24+24=80) to D_spatial
        self.final_mlp = nn.Sequential(
            nn.Linear(32 + 24 + 24, 128),
            nn.ReLU(),
            nn.Linear(128, D_spatial)
        )
        
        # Normalization for RGB using ImageNet stats
        self.rgb_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.rgb_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        
        self.to(self.device)

    def forward(self, skeletal, crops, optical_flow):
        B, T, _, _, _ = skeletal.shape
        
        # Flatten inputs: merge batch and time*hand dimensions (e.g., left/right)
        skeletal_flat = skeletal.reshape(B*T*2, 21, 3).to(self.device)
        crops_flat = crops.reshape(B*T*2, 3, 112, 112).to(self.device)
        flow_flat = optical_flow.reshape(B*T*2, 2, 112, 112).to(self.device)
        
        # Normalize RGB inputs
        crops_flat = (crops_flat - self.rgb_mean) / self.rgb_std
        
        # -------------------------
        # First Block Processing
        # -------------------------
        skeletal_feat1 = self.graph_layer1(skeletal_flat)  # (B*T*2, 21, 32)
        rgb_feat1 = self.first_block_rgb(crops_flat)         # (B*T*2, 16, H1, W1)
        flow_feat1 = self.first_block_flow(flow_flat)        # (B*T*2, 16, H2, W2)
        
        # Pool features for fusion: average pooling over nodes or spatial dims
        skeletal_pooled1 = skeletal_feat1.mean(dim=1)         # (B*T*2, 32)
        rgb_pooled1 = rgb_feat1.mean(dim=(2, 3))                # (B*T*2, 16)
        flow_pooled1 = flow_feat1.mean(dim=(2, 3))              # (B*T*2, 16)
        
        # Cross-modal fusion block 1 with our new SE-based fusion
        fused1 = self.fusion_block1([skeletal_pooled1, rgb_pooled1, flow_pooled1])
        skeletal_fused1, rgb_fused1, flow_fused1 = fused1
        
        # Residual update (with unsqueeze to match spatial dims)
        skeletal_feat1 = skeletal_feat1 + skeletal_fused1.unsqueeze(1)
        rgb_feat1 = rgb_feat1 + rgb_fused1.unsqueeze(2).unsqueeze(3)
        flow_feat1 = flow_feat1 + flow_fused1.unsqueeze(2).unsqueeze(3)
        
        # -------------------------
        # Second Block Processing
        # -------------------------
        skeletal_feat2 = self.graph_layer2(skeletal_feat1)     # (B*T*2, 21, 32)
        rgb_feat2 = self.second_block_rgb(rgb_feat1)           # (B*T*2, 24, H3, W3)
        flow_feat2 = self.second_block_flow(flow_feat1)          # (B*T*2, 24, H4, W4)
        
        # Pool features for fusion in second block
        skeletal_final = skeletal_feat2.mean(dim=1)            # (B*T*2, 32)
        rgb_final = rgb_feat2.mean(dim=(2, 3))                   # (B*T*2, 24)
        flow_final = flow_feat2.mean(dim=(2, 3))                 # (B*T*2, 24)
        
        # Cross-modal fusion block 2 with our new SE-based fusion
        fused2 = self.fusion_block2([skeletal_final, rgb_final, flow_final])
        skeletal_fused2, rgb_fused2, flow_fused2 = fused2
        
        # Residual update with the fused signals
        skeletal_final = skeletal_final + skeletal_fused2
        rgb_final = rgb_final + rgb_fused2
        flow_final = flow_final + flow_fused2
        
        # -------------------------
        # Final Fusion
        # -------------------------
        final_concat = torch.cat([skeletal_final, rgb_final, flow_final], dim=1)  # (B*T*2, 80)
        spatial_features = self.final_mlp(final_concat)                          # (B*T*2, D_spatial)
        
        # Reshape back to (B, T, 2, D_spatial)
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
