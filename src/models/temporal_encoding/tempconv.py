import torch
import torch.nn as nn
import math

# Depthwise Separable Convolution
class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, padding=0):
        super(DepthwiseSeparableConv1d, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, groups=in_channels, dilation=dilation, padding=padding)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# Efficient Channel Attention with Learnable Convolution
def get_kernel_size(channel):
    k = int(abs((math.log2(channel) + 1) / 2))
    return k if k % 2 else k + 1

class ECA(nn.Module):
    def __init__(self, channel, device='cpu'):
        super(ECA, self).__init__()
        self.k_size = get_kernel_size(channel)
        self.conv = nn.Conv1d(1, 1, kernel_size=self.k_size, padding=(self.k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, T)
        y = x.mean(dim=2, keepdim=True)  # (B, C, 1)
        y = y.transpose(1, 2)  # (B, 1, C)
        y = self.conv(y)  # (B, 1, C)
        y = y.transpose(1, 2)  # (B, C, 1)
        y = self.sigmoid(y)
        return x * y

# Optimized Multi-Scale Temporal Convolution
class MultiScaleTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=[1, 2, 4, 8], kernel_size=3, device='cpu'):
        super(MultiScaleTemporalConv, self).__init__()
        self.device = torch.device(device)
        num_branches = len(dilations)
        out_channels_per_branch = out_channels // num_branches
        self.branches = nn.ModuleList()
        for d in dilations:
            padding = (kernel_size - 1) * d // 2
            self.branches.append(
                nn.Sequential(
                    DepthwiseSeparableConv1d(in_channels, out_channels_per_branch, kernel_size, dilation=d, padding=padding),
                    nn.BatchNorm1d(out_channels_per_branch),
                    nn.ReLU(inplace=True)
                )
            )
        self.eca = ECA(out_channels).to(device)

    def forward(self, x):
        x = x.to(self.device)
        branch_outputs = [branch(x) for branch in self.branches]
        concatenated = torch.cat(branch_outputs, dim=1)  # (B*2, out_channels, T)
        calibrated = self.eca(concatenated)
        return calibrated

# Residual Block
class ResidualTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=[1, 2, 4, 8], kernel_size=3, device='cpu'):
        super(ResidualTemporalConv, self).__init__()
        self.multi_scale_conv = MultiScaleTemporalConv(in_channels, out_channels, dilations, kernel_size, device)
        self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1).to(device) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual_conv(x)
        x = self.multi_scale_conv(x)
        return x + residual

# Temporal Encoding Module
class TemporalEncoding(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2, dilations=[1, 2, 4, 8], kernel_size=3, device='cpu'):
        super(TemporalEncoding, self).__init__()
        self.device = torch.device(device)
        layers = []
        for _ in range(num_layers):
            layers.append(
                ResidualTemporalConv(in_channels, out_channels, dilations, kernel_size, device)
            )
            in_channels = out_channels
        self.temporal_conv = nn.Sequential(*layers).to(device)

    def forward(self, x):
        x = x.to(self.device)
        B, T, num_hands, D_total = x.shape
        x = x.view(B * num_hands, D_total, T)
        x = self.temporal_conv(x)
        x = x.view(B, num_hands, -1, T).permute(0, 3, 1, 2)
        return x

# Example Usage
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(4, 191, 2, 1088).to(device)
    model = TemporalEncoding(in_channels=1088, out_channels=256, num_layers=2, device=device)
    output = model(x)
    print(f"Input shape: {x.shape}")  # [4, 191, 2, 1088]
    print(f"Output shape: {output.shape}")  # [4, 191, 2, 256]