import torch
import torch.nn as nn

# Depthwise Separable Convolution (unchanged)
class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, padding=0):
        super(DepthwiseSeparableConv1d, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, groups=in_channels, dilation=dilation, padding=padding)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# Residual Depthwise Separable Convolution Block
class ResidualDepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, device='cpu'):
        super(ResidualDepthwiseSeparableConv1d, self).__init__()
        self.conv = DepthwiseSeparableConv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            dilation=dilation, 
            padding=(kernel_size - 1) * dilation // 2  # Preserve sequence length
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # Residual projection if channels differ
        self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1).to(device) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual_conv(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x + residual

# Modified Temporal Encoding Module
class TemporalEncoding(nn.Module):
    def __init__(self, in_channels, out_channels=256, kernel_size=3, dilations=[1, 2, 4], vocab_size=None, device='cpu'):
        super(TemporalEncoding, self).__init__()
        self.device = torch.device(device)
        
        # Stack 3 layers with specified dilations
        self.layers = nn.ModuleList([
            ResidualDepthwiseSeparableConv1d(
                in_channels if i == 0 else out_channels, 
                out_channels, 
                kernel_size, 
                dilation=d, 
                device=device
            )
            for i, d in enumerate(dilations)
        ])
        
        # Auxiliary CTC head (optional, enabled if vocab_size is provided)
        if vocab_size is not None:
            self.aux_conv = nn.Conv1d(out_channels, 64, kernel_size=3, padding=1).to(device)
            self.aux_linear = nn.Linear(64, vocab_size + 1).to(device)  # +1 for CTC blank token
        else:
            self.aux_conv = None

    def forward(self, x):
        x = x.to(self.device)
        B, T, num_hands, D_spatial = x.shape  # e.g., (B, T, 2, 128)
        x = x.view(B * num_hands, D_spatial, T)  # (B*2, D_spatial, T)

        # First layer with auxiliary head
        x = self.layers[0](x)  # (B*2, 256, T)
        if self.aux_conv is not None:
            # Reshape and average over hands
            x_aux = x.view(B, num_hands, 256, T).mean(dim=1)  # (B, 256, T)
            x_aux = self.aux_conv(x_aux)  # (B, 64, T)
            x_aux = x_aux.permute(0, 2, 1)  # (B, T, 64)
            aux_output = self.aux_linear(x_aux)  # (B, T, vocab_size + 1)
        else:
            aux_output = None

        # Remaining layers
        for layer in self.layers[1:]:
            x = layer(x)  # (B*2, 256, T)

        # Reshape back to (B, T, num_hands, out_channels)
        x = x.view(B, num_hands, 256, T).permute(0, 3, 1, 2)  # (B, T, 2, 256)
        return x, aux_output

# Example Usage
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 15  # Example gloss vocabulary size
    model = TemporalEncoding(
        in_channels=128,  # Assuming D_spatial=128 from spatial encoding
        out_channels=256,
        kernel_size=3,
        dilations=[1, 2, 4],
        vocab_size=vocab_size,
        device=device
    )
    x = torch.randn(4, 114, 2, 128).to(device)  # Example input
    output, aux_output = model(x)
    print(f"Input shape: {x.shape}")  # [4, 114, 2, 128]
    print(f"Output shape: {output.shape}")  # [4, 114, 2, 256]
    if aux_output is not None:
        print(f"Auxiliary output shape: {aux_output.shape}")  # [4, 114, 101]