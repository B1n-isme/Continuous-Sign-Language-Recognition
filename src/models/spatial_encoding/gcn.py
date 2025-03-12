import torch
import torch.nn as nn


class Graph:
    """Simple hand graph with adjacency matrix."""

    def __init__(self):
        self.num_nodes = 21
        # Define adjacency matrix A (21x21) based on hand topology
        # Example: Connect fingertips to joints, joints to palm (simplified as identity + basic connections)
        self.A = torch.eye(21)  # Placeholder; replace with real hand graph


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super(STGCNBlock, self).__init__()
        self.gcn = nn.Conv2d(
            in_channels, out_channels, kernel_size=1
        )  # Spatial graph convolution
        self.tcn = nn.Conv2d(
            out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)
        )  # Temporal conv
        self.A = A  # Adjacency matrix
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (B, C, T, V) where V=21 (vertices)
        x = self.gcn(x)  # Spatial convolution
        # Fix: Corrected einsum equation to match tensor dimensions
        x = x + torch.einsum("nctv,vw->nctw", (x, self.A))  # Graph multiplication
        x = self.tcn(x)  # Temporal convolution
        x = self.bn(x)
        x = self.relu(x)
        return x


class STGCN(nn.Module):
    def __init__(self, in_channels=3, num_joints=21, out_dim=64, num_layers=2):
        super(STGCN, self).__init__()
        self.graph = Graph()
        # Fix: Use clone().detach() instead of torch.tensor()
        A = self.graph.A.clone().detach().requires_grad_(False)
        self.register_buffer("A", A)

        self.layers = nn.ModuleList()
        self.layers.append(STGCNBlock(in_channels, 32, A))
        for _ in range(num_layers - 1):
            self.layers.append(STGCNBlock(32, 32, A))

        self.fc = nn.Linear(32 * num_joints, out_dim)  # Reduce to D_skeletal

    def forward(self, x):
        # x: (B, T, 21, 3)
        x = x.permute(0, 3, 1, 2)  # (B, C=3, T, V=21)
        for layer in self.layers:
            x = layer(x)  # (B, 32, T, 21)
        x = x.permute(0, 2, 1, 3).reshape(x.size(0), x.size(2), -1)  # (B, T, 32*21)
        x = self.fc(x)  # (B, T, 64)
        return x
