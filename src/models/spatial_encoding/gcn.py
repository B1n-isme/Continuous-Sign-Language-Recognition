import torch
import torch.nn as nn

class Graph:
    """Hand graph with adjacency matrix based on topology."""
    def __init__(self, num_nodes=21, device='cpu'):
        self.num_nodes = num_nodes
        self.device = torch.device(device)
        self.A = self.build_hand_adjacency()

    def build_hand_adjacency(self):
        # Define hand joint connections (0: wrist, 1-4: thumb, 5-8: index, etc.)
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),      # Index
            (0, 9), (9, 10), (10, 11), (11, 12), # Middle
            (0, 13), (13, 14), (14, 15), (15, 16), # Ring
            (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
        ]
        A = torch.zeros(self.num_nodes, self.num_nodes)
        for i, j in connections:
            A[i, j] = 1
            A[j, i] = 1  # Undirected graph
        
        # Add self-loops
        A = A + torch.eye(self.num_nodes)
        
        # Symmetric normalization: D^-0.5 * A * D^-0.5
        D = torch.diag(A.sum(1) ** -0.5)
        A = D @ A @ D
        return A

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, device='cpu'):
        super(STGCNBlock, self).__init__()
        self.device = torch.device(device)
        self.A = A  # (V, V)
        
        # Spatial graph convolution (1x1 conv for feature transformation)
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=1).to(self.device)
        
        # Temporal convolution
        self.tcn = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), 
                            padding=(1, 0), stride=(stride, 1)).to(self.device)
        
        self.bn = nn.BatchNorm2d(out_channels).to(self.device)
        self.relu = nn.ReLU(inplace=True).to(self.device)
        
        # Residual connection
        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)).to(self.device),
                nn.BatchNorm2d(out_channels).to(self.device)
            )
        else:
            self.residual = nn.Identity().to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        # x: (B, C, T, V) e.g., (B, 2, T, 21)
        res = self.residual(x.contiguous())  # Ensure contiguous tensor
        
        # Spatial graph convolution: A @ x
        x = torch.einsum("vu, bctu -> bctv", self.A, x)  # (B, C, T, V)
        x = self.gcn(x)  # (B, out_channels, T, V)
        
        # Temporal convolution
        x = self.tcn(x)  # (B, out_channels, T, V)
        x = self.bn(x)
        
        # Add residual
        x = x + res
        x = self.relu(x)
        return x

class STGCN(nn.Module):
    def __init__(self, in_channels=2, num_joints=21, out_dim=64, num_layers=2, device='cpu'):
        super(STGCN, self).__init__()
        self.device = torch.device(device)
        self.graph = Graph(num_joints, device=device)
        A = self.graph.A.clone().detach().requires_grad_(False).to(self.device)
        self.register_buffer("A", A)

        # Define layers for both hands
        layers = []
        for i in range(num_layers):
            in_c = in_channels if i == 0 else 32
            layers.append(STGCNBlock(in_c, 32, A, device=device))
        self.stgcn = nn.ModuleList(layers).to(self.device)

        # Final fully connected layer
        self.fc = nn.Linear(32 * num_joints, out_dim).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        # x: (B, T, 2, 21, 2) - 2D keypoints for left and right hands
        B, T, H, V, C = x.shape  # H=2 (hands), V=21 (joints), C=2 (x, y)
        left_hand = x[:, :, 0, :, :].permute(0, 3, 1, 2).to(self.device)  # (B, 2, T, 21)
        right_hand = x[:, :, 1, :, :].permute(0, 3, 1, 2).to(self.device) # (B, 2, T, 21)

        # Process each hand through STGCN layers
        for layer in self.stgcn:
            left_hand = layer(left_hand)    # (B, 32, T, 21)
            right_hand = layer(right_hand)  # (B, 32, T, 21)

        # Reshape and reduce
        left_hand = left_hand.permute(0, 2, 1, 3).reshape(B, T, -1).to(self.device)  # (B, T, 32*21)
        right_hand = right_hand.permute(0, 2, 1, 3).reshape(B, T, -1).to(self.device) # (B, T, 32*21)

        left_features = self.fc(left_hand)   # (B, T, 64)
        right_features = self.fc(right_hand) # (B, T, 64)

        # Stack features for both hands
        features = torch.stack([left_features, right_features], dim=2)  # (B, T, 2, 64)
        return features

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize model and move to CUDA
    model = STGCN(in_channels=2, num_joints=21, out_dim=64, num_layers=2, device=device).to(device)
    
    # Example input tensor
    x = torch.randn(4, 191, 2, 21, 2).to(device)  # (B, T, 2, 21, 2)
    
    # Forward pass
    out = model(x)
    
    print(f"Input shape: {x.shape}")   # Expected: (4, 191, 2, 21, 2)
    print(f"Output shape: {out.shape}") # Expected: (4, 191, 2, 64)