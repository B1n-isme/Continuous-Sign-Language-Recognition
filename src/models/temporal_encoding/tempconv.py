import torch
import torch.nn as nn

class MultiScaleTemporalConv(nn.Module):
    """Multi-scale temporal convolution with different kernel sizes and dilations."""
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7], dilations=[1, 2, 4]):
        super(MultiScaleTemporalConv, self).__init__()
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            for d in dilations:
                padding = (k - 1) * d // 2  # Same padding to maintain sequence length
                self.branches.append(
                    nn.Sequential(
                        nn.Conv1d(in_channels, out_channels, kernel_size=k, dilation=d, padding=padding),
                        nn.BatchNorm1d(out_channels),
                        nn.ReLU(inplace=True)
                    )
                )
        
        # 1x1 conv to reduce dimensionality after concatenation
        self.reduce_conv = nn.Conv1d(len(kernel_sizes) * len(dilations) * out_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Input: (B*2, D_total, T)
        branch_outputs = [branch(x) for branch in self.branches]
        concatenated = torch.cat(branch_outputs, dim=1)  # (B*2, num_branches * out_channels, T)
        reduced = self.reduce_conv(concatenated)  # (B*2, out_channels, T)
        return reduced

class ResidualTemporalConv(nn.Module):
    """Residual block with multi-scale temporal convolution."""
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7], dilations=[1, 2, 4]):
        super(ResidualTemporalConv, self).__init__()
        self.multi_scale_conv = MultiScaleTemporalConv(in_channels, out_channels, kernel_sizes, dilations)
        self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        residual = self.residual_conv(x)  # Match dimensions if needed
        x = self.multi_scale_conv(x)
        x = x + residual  # Add residual connection
        return x

class TemporalEncoding(nn.Module):
    """Temporal encoding module with stacked residual multi-scale convolutions."""
    def __init__(self, in_channels, out_channels, num_layers=2, kernel_sizes=[3, 5, 7], dilations=[1, 2, 4]):
        super(TemporalEncoding, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(
                ResidualTemporalConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_sizes=kernel_sizes,
                    dilations=dilations
                )
            )
            in_channels = out_channels  # Update for next layer
        self.temporal_conv = nn.Sequential(*layers)
    
    def forward(self, x):
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
    # Sample input: batch_size=4, sequence_length=104, num_hands=2, D_total=1088
    x = torch.randn(4, 191, 2, 1088)
    model = TemporalEncoding(in_channels=1088, out_channels=256, num_layers=2)
    output = model(x)
    print(f"Input shape: {x.shape}")  # torch.Size([4, 104, 2, 1088])
    print(f"Output shape: {output.shape}")  # torch.Size([4, 104, 2, 256])