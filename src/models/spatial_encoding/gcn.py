import torch
import torch.nn as nn

class Graph:
    """Hand graph with adjacency matrix based on topology."""
    def __init__(self, num_nodes=21):
        self.num_nodes = num_nodes
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
    def __init__(self, in_channels, out_channels, A, stride=1):
        super(STGCNBlock, self).__init__()
        self.A = A  # (V, V)
        
        # Spatial graph convolution (1x1 conv for feature transformation)
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # Temporal convolution
        self.tcn = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), 
                            padding=(1, 0), stride=(stride, 1))
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual connection
        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                    stride=(stride, 1))
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        # x: (B, C, T, V)
        res = self.residual(x)
        
        # Spatial graph convolution: A @ x
        x = torch.einsum("vu, bctu -> bctv", (self.A, x))
        x = self.gcn(x)
        
        # Temporal convolution
        x = self.tcn(x)
        x = self.bn(x)
        
        # Add residual
        x = x + res
        x = self.relu(x)
        return x

class STGCN(nn.Module):
    def __init__(self, in_channels=2, num_joints=21, out_dim=64, num_layers=2):
        super(STGCN, self).__init__()
        self.graph = Graph(num_joints)
        A = self.graph.A.clone().detach().requires_grad_(False)
        self.register_buffer("A", A)

        # Separate STGCNs for left and right hands
        self.stgcn_left = nn.ModuleList([STGCNBlock(in_channels, 32, A) if i == 0 else 
                                        STGCNBlock(32, 32, A) for i in range(num_layers)])
        self.stgcn_right = nn.ModuleList([STGCNBlock(in_channels, 32, A) if i == 0 else 
                                         STGCNBlock(32, 32, A) for i in range(num_layers)])

        self.fc_left = nn.Linear(32 * num_joints, out_dim)
        self.fc_right = nn.Linear(32 * num_joints, out_dim)

    def forward(self, x):
        # x: (B, T, 2, 21, 2) - 2D keypoints for left and right hands
        left_hand = x[:, :, 0, :, :]  # (B, T, 21, 2)
        right_hand = x[:, :, 1, :, :] # (B, T, 21, 2)

        # Permute to (B, C, T, V)
        left_hand = left_hand.permute(0, 3, 1, 2)  # (B, 2, T, 21)
        right_hand = right_hand.permute(0, 3, 1, 2) # (B, 2, T, 21)

        # Process each hand
        for layer in self.stgcn_left:
            left_hand = layer(left_hand)  # (B, 32, T, 21)
        for layer in self.stgcn_right:
            right_hand = layer(right_hand) # (B, 32, T, 21)

        # Reshape and reduce
        left_hand = left_hand.permute(0, 2, 1, 3).reshape(left_hand.size(0), left_hand.size(2), -1)  # (B, T, 32*21)
        right_hand = right_hand.permute(0, 2, 1, 3).reshape(right_hand.size(0), right_hand.size(2), -1) # (B, T, 32*21)

        left_features = self.fc_left(left_hand)   # (B, T, 64)
        right_features = self.fc_right(right_hand) # (B, T, 64)

        # Combine features (e.g., concatenate)
        features = torch.cat([left_features, right_features], dim=-1)  # (B, T, 128)
        return features