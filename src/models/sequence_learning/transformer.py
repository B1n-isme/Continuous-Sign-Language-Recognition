import torch
import torch.nn as nn
import math
from performer_pytorch import SelfAttention

# Positional Encoding (unchanged)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :].to(x.device)
        return x

# Depthwise Separable Feedforward Network (unchanged)
class DSFeedforward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(DSFeedforward, self).__init__()
        self.depthwise = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.pointwise = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.pointwise2 = nn.Conv1d(d_ff, d_model, kernel_size=1)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, d_model, T)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.pointwise2(x)
        x = x.transpose(1, 2)  # (B, T, d_model)
        return x

# Custom Transformer Encoder Layer with Performer Attention
class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(CustomTransformerEncoderLayer, self).__init__()
        # Replace nn.MultiheadAttention with PerformerAttention
        self.self_attn = SelfAttention(
            dim=d_model,
            heads=nhead,
            dim_head=d_model // nhead,  # Ensure divisibility
            causal=False,               # Bidirectional for encoder
            dropout=dropout
        )
        self.feed_forward = DSFeedforward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Pre-Layer Normalization
        src2 = self.norm1(src)
        # PerformerAttention expects (B, T, D) and returns (B, T, D)
        attn_output = self.self_attn(src2)
        src = src + self.dropout(attn_output)
        src2 = self.norm2(src)
        src = src + self.dropout(self.feed_forward(src2))
        return src

# Transformer Sequence Learning with Performer Attention
class TransformerSequenceLearning(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, vocab_size, dropout=0.1, device="cpu"):
        super(TransformerSequenceLearning, self).__init__()
        self.device = torch.device(device)
        self.model_dim = model_dim

        # Input projection to reduced dimension
        self.input_proj = nn.Linear(input_dim, model_dim).to(device)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(model_dim).to(device)

        # Transformer encoder with Performer-based layers
        encoder_layers = [CustomTransformerEncoderLayer(model_dim, num_heads, model_dim * 2, dropout).to(device)
                          for _ in range(num_layers)]
        self.transformer_encoder = nn.ModuleList(encoder_layers)

        # Classification head
        self.classifier = nn.Linear(model_dim, vocab_size).to(device)

    def forward(self, x):
        x = x.to(self.device)
        B, T, num_hands, D = x.shape
        x = x.view(B, T, -1)  # (B, T, 2*D)
        x = self.input_proj(x)  # (B, T, model_dim)
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)  # Add positional encoding

        # Pass through Performer-based Transformer layers
        for layer in self.transformer_encoder:
            x = layer(x)

        gloss_probs = self.classifier(x)  # (B, T, vocab_size)
        return gloss_probs

# Example Usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(4, 191, 2, 256).to(device)
    model = TransformerSequenceLearning(
        input_dim=2 * 256,  # 2 * 256 = 512
        model_dim=128,      # Reduced model dimension
        num_heads=4,        # Multi-head attention
        num_layers=2,       # Lightweight with 2 layers
        vocab_size=10,      # Example vocab size
        dropout=0.1,
        device=device
    ).to(device)
    output = model(x)
    print(f"Input shape: {x.shape}")  # [4, 191, 2, 256]
    print(f"Output shape: {output.shape}")  # [4, 191, 10]