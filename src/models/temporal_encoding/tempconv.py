import torch
import torch.nn as nn
import torch.nn.functional as F

# Depthwise Separable Convolution (unchanged)
class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, padding=0):
        super(DepthwiseSeparableConv1d, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size,
                                   groups=in_channels, dilation=dilation, padding=padding)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# Residual Depthwise Separable Convolution Block with GroupNorm
class ResidualDepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(ResidualDepthwiseSeparableConv1d, self).__init__()
        padding = (kernel_size - 1) * dilation // 2  # Preserve sequence length
        self.conv = DepthwiseSeparableConv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, padding=padding
        )
        # Replace BatchNorm1d with GroupNorm (using 16 groups)
        self.gn = nn.GroupNorm(num_groups=16, num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)
        # Residual projection if channels differ
        self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual_conv(x)
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x + residual

# Multi-Scale Residual Block that fuses different kernel sizes
class MultiScaleResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, dilation):
        """
        kernel_sizes: list of kernel sizes (e.g., [3, 5, 7])
        dilation: dilation factor for this block
        """
        super(MultiScaleResidualBlock, self).__init__()
        self.branches = nn.ModuleList([
            ResidualDepthwiseSeparableConv1d(
                in_channels, out_channels, kernel_size, dilation=dilation
            ) for kernel_size in kernel_sizes
        ])
        # Fuse concatenated features (len(kernel_sizes)*out_channels -> out_channels)
        self.fuse_conv = nn.Conv1d(len(kernel_sizes)*out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Run each branch in parallel
        branch_outputs = [branch(x) for branch in self.branches]
        # Concatenate along channel dimension
        x_cat = torch.cat(branch_outputs, dim=1)
        # Fuse with a 1x1 convolution
        fused = self.fuse_conv(x_cat)
        return fused

# Modified Temporal Encoding Module using Multi-Scale Residual Blocks with device support
class TemporalEncoding(nn.Module):
    def __init__(self, in_channels, out_channels=256,
                 kernel_sizes=[3, 5, 7], dilations=[1, 2, 4],
                 vocab_size=None, device='cpu'):
        super(TemporalEncoding, self).__init__()
        self.device = torch.device(device)
        
        # Build a stack of multi-scale residual blocks
        layers = []
        for i, dilation in enumerate(dilations):
            block = MultiScaleResidualBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_sizes,
                dilation
            )
            layers.append(block.to(self.device))
        self.layers = nn.ModuleList(layers)
        
        # Auxiliary CTC head (optional)
        if vocab_size is not None:
            self.aux_conv = nn.Conv1d(out_channels, 64, kernel_size=3, padding=1).to(self.device)
            self.aux_linear = nn.Linear(64, vocab_size).to(self.device)
        else:
            self.aux_conv = None

    def forward(self, x):
        # x: (B, T, num_hands, D_spatial), e.g., (B, T, 2, 128)
        x = x.to(self.device)
        B, T, num_hands, D_spatial = x.shape
        # Merge num_hands with batch for 1D processing: (B*num_hands, D_spatial, T)
        x = x.view(B * num_hands, D_spatial, T)

        # Process through multi-scale residual blocks
        # Use first block for auxiliary CTC head if available
        x = self.layers[0](x)  # (B*num_hands, out_channels, T)
        if self.aux_conv is not None:
            # Reshape to (B, num_hands, out_channels, T) and average over hands
            x_aux = x.view(B, num_hands, -1, T).mean(dim=1)  # (B, out_channels, T)
            x_aux = self.aux_conv(x_aux)  # (B, 64, T)
            x_aux = x_aux.permute(0, 2, 1)  # (B, T, 64)
            aux_output = self.aux_linear(x_aux)  # (B, T, vocab_size)
        else:
            aux_output = None

        # Process remaining multi-scale blocks
        for layer in self.layers[1:]:
            x = layer(x)  # (B*num_hands, out_channels, T)

        # Reshape back to (B, T, num_hands, out_channels)
        x = x.view(B, num_hands, -1, T).permute(0, 3, 1, 2)  # (B, T, num_hands, out_channels)
        return x, aux_output

# Example Usage
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 15  # Example gloss vocabulary size
    model = TemporalEncoding(
        in_channels=128,   # Assuming D_spatial=128 from spatial encoding
        out_channels=256,
        kernel_sizes=[3, 5, 7],
        dilations=[1, 2, 4],
        vocab_size=vocab_size,
        device=device
    ).to(device)
    
    x = torch.randn(4, 114, 2, 128).to(device)  # Example input: (B, T, 2, 128)
    output, aux_output = model(x)
    print(f"Input shape: {x.shape}")           # Expected: [4, 114, 2, 128]
    print(f"Output shape: {output.shape}")       # Expected: [4, 114, 2, 256]
    if aux_output is not None:
        print(f"Auxiliary output shape: {aux_output.shape}")  # Expected: [4, 114, vocab_size]
