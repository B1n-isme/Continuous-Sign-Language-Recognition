import torch
import torch.nn as nn

# ECA Module (Efficient Channel Attention)
class ECA(nn.Module):
    def __init__(self, gamma=2, b=1, device='cpu'):
        super(ECA, self).__init__()
        self.gamma = gamma
        self.b = b
        self.device = torch.device(device)

    def forward(self, x):
        x = x.to(self.device)
        
        # Global average pooling: [B, C, T] -> [B, C, 1]
        y = torch.mean(x, dim=2, keepdim=True)

        # Adaptive kernel size calculation
        t = int(abs((torch.log(torch.tensor(x.size(1), dtype=torch.float32, device=self.device)) / 
             torch.log(torch.tensor(2.0, dtype=torch.float32, device=self.device)) + self.b) / self.gamma))

        kernel_size = t if t % 2 else t + 1

        # Apply grouped 1D convolution with dynamic kernel size
        y = torch.nn.functional.conv1d(
            y,
            weight=torch.ones(x.size(1), 1, kernel_size, device=x.device) / kernel_size,
            groups=x.size(1),
            padding=kernel_size // 2,
        )
        return x * y

class MultiScaleTemporalConv(nn.Module):
    """Multi-scale temporal convolution with different kernel sizes and dilations."""
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7], dilations=[1, 2, 4], device='cpu'):
        super(MultiScaleTemporalConv, self).__init__()
        self.device = torch.device(device)
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            for d in dilations:
                padding = (k - 1) * d // 2  # Same padding to maintain sequence length
                self.branches.append(
                    nn.Sequential(
                        nn.Conv1d(in_channels, out_channels, kernel_size=k, dilation=d, padding=padding).to(device),
                        nn.BatchNorm1d(out_channels).to(device),
                        nn.ReLU(inplace=True).to(device)
                    )
                )
        # 1x1 conv to reduce dimensionality after concatenation
        self.reduce_conv = nn.Conv1d(len(kernel_sizes) * len(dilations) * out_channels, out_channels, kernel_size=1).to(device)
        # Apply ECA after concatenation to recalibrate channel responses
        self.eca = ECA().to(device)

    def forward(self, x):
        x = x.to(self.device)
        # Input: (B*2, D_total, T)
        branch_outputs = [branch(x) for branch in self.branches]
        concatenated = torch.cat(branch_outputs, dim=1)  # (B*2, num_branches * out_channels, T)
        
        # Apply ECA to the concatenated features
        calibrated = self.eca(concatenated)
        
        # Dimensionality reduction via 1x1 conv
        reduced = self.reduce_conv(calibrated)  # (B*2, out_channels, T)
        return reduced

class ResidualTemporalConv(nn.Module):
    """Residual block with multi-scale temporal convolution."""
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7], dilations=[1, 2, 4], device='cpu'):
        super(ResidualTemporalConv, self).__init__()
        self.device = torch.device(device)
        self.multi_scale_conv = MultiScaleTemporalConv(in_channels, out_channels, kernel_sizes, dilations, device=device).to(device)
        self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1).to(device) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        x = x.to(self.device)
        residual = self.residual_conv(x)  # Match dimensions if needed
        x = self.multi_scale_conv(x)
        x = x + residual  # Add residual connection
        return x

class TemporalEncoding(nn.Module):
    """Temporal encoding module with stacked residual multi-scale convolutions."""
    def __init__(self, in_channels, out_channels, num_layers=2, kernel_sizes=[3, 5, 7], dilations=[1, 2, 4], device='cpu'):
        super(TemporalEncoding, self).__init__()
        self.device = torch.device(device)
        layers = []
        for _ in range(num_layers):
            layers.append(
                ResidualTemporalConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_sizes=kernel_sizes,
                    dilations=dilations,
                    device=device
                )
            )
            in_channels = out_channels  # Update for next layer
        self.temporal_conv = nn.Sequential(*layers).to(device)

    def forward(self, x):
        x = x.to(self.device)
        # Input shape: (B, T, 2, D_total)
        B, T, num_hands, D_total = x.shape
        
        # Reshape to (B*num_hands, D_total, T)
        x = x.view(B * num_hands, D_total, T)
        
        # Apply temporal convolutions
        x = self.temporal_conv(x)  # (B*num_hands, D_temp, T)
        
        # Reshape back to (B, T, 2, D_temp)
        x = x.view(B, num_hands, -1, T).permute(0, 3, 1, 2)  # (B, T, 2, D_temp)
        
        return x

# Example usage (for testing)
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Sample input: batch_size=4, sequence_length=191, num_hands=2, D_total=1088
    x = torch.randn(4, 191, 2, 1088).to(device)
    model = TemporalEncoding(in_channels=1088, out_channels=256, num_layers=2, device=device).to(device)
    output = model(x)
    print(f"Input shape: {x.shape}")  # torch.Size([4, 191, 2, 1088])
    print(f"Output shape: {output.shape}")  # torch.Size([4, 191, 2, 256])
