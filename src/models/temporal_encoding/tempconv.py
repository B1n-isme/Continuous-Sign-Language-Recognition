import torch
import torch.nn as nn

class TemporalConvBlock(nn.Module):
    """A single 1D convolutional block with convolution, batch norm, and ReLU."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(TemporalConvBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class TemporalConv(nn.Module):
    """A stack of TemporalConvBlocks to process temporal sequences."""
    def __init__(self, in_channels, out_channels, num_layers=2, kernel_size=3):
        super(TemporalConv, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(
                TemporalConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size
                )
            )
            in_channels = out_channels  # Update in_channels for the next layer
        self.temporal_conv = nn.Sequential(*layers)

    def forward(self, x):
        # Input: (B*2, D_total, T)
        x = self.temporal_conv(x)  # Output: (B*2, D_temp, T)
        return x

class TemporalEncoding(nn.Module):
    """Temporal encoding module for processing spatially encoded features."""
    def __init__(self, in_channels, out_channels, num_layers=2, kernel_size=3):
        super(TemporalEncoding, self).__init__()
        self.temporal_conv = TemporalConv(
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            kernel_size=kernel_size
        )

    def forward(self, x):
        # Input shape: (B, T, 2, D_total)
        B, T, num_hands, D_total = x.shape
        
        # Reshape to (B*num_hands, D_total, T) for per-hand processing
        x = x.view(B * num_hands, D_total, T)
        
        # Apply temporal convolutions
        x = self.temporal_conv(x)  # Shape: (B*num_hands, D_temp, T)
        
        # Reshape back to (B, T, 2, D_temp)
        x = x.view(B, num_hands, -1, T).permute(0, 3, 1, 2)
        
        return x  # Output shape: (B, T, 2, D_temp)

# Example usage (for testing)
if __name__ == "__main__":
    # Sample input: batch_size=4, sequence_length=191, num_hands=2, D_total=1088
    x = torch.randn(4, 191, 2, 1088)
    model = TemporalEncoding(in_channels=1088, out_channels=256)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")  # Should be (4, 191, 2, 256)