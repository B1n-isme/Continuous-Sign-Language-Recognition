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

# Custom Transformer Encoder Layer with Performer Attention
class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(CustomTransformerEncoderLayer, self).__init__()
        # Use Performer SelfAttention
        self.self_attn = SelfAttention(
            dim=d_model,
            heads=nhead,
            dim_head=d_model // nhead,  # Ensure divisibility (256 / 4 = 64)
            causal=False,               # Bidirectional for encoder
            dropout=dropout
        )
        # Standard feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Pre-Layer Normalization
        src2 = self.norm1(src)
        # Performer Attention expects (B, T, D) and returns (B, T, D)
        attn_output = self.self_attn(src2)
        src = src + self.dropout1(attn_output)
        # Feedforward
        src2 = self.norm2(src)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(ff_output)
        return src

# Updated Transformer Sequence Learning with Performer Attention
class TransformerSequenceLearning(nn.Module):
    def __init__(self, input_dim, model_dim=256, num_heads=4, num_layers=2, vocab_size=None, dropout=0.1, device="cpu"):
        super(TransformerSequenceLearning, self).__init__()
        self.device = torch.device(device)
        self.model_dim = model_dim

        # Input projection to model_dim (256)
        self.input_proj = nn.Linear(input_dim, model_dim).to(device)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(model_dim).to(device)

        # Transformer encoder with Performer-based layers
        encoder_layers = [CustomTransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,  # Standard FFN size (256 * 4 = 1024)
            dropout=dropout
        ).to(device) for _ in range(num_layers)]
        self.transformer_encoder = nn.ModuleList(encoder_layers)

        # Main classification head
        self.classifier = nn.Linear(model_dim, vocab_size).to(device)

        # Auxiliary CTC head
        self.aux_conv = nn.Conv1d(model_dim, 64, kernel_size=3, padding=1).to(device)
        self.aux_linear = nn.Linear(64, vocab_size + 1).to(device)  # +1 for CTC blank token

    def forward(self, x):
        x = x.to(self.device)
        B, T, num_hands, D = x.shape
        x = x.view(B, T, -1)  # (B, T, num_hands * D)
        x = self.input_proj(x)  # (B, T, model_dim)
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)  # Add positional encoding

        # First Transformer layer with auxiliary CTC head
        x = self.transformer_encoder[0](x)
        # Auxiliary output
        x_aux = x.permute(0, 2, 1)  # (B, model_dim, T)
        x_aux = self.aux_conv(x_aux)  # (B, 64, T)
        x_aux = x_aux.permute(0, 2, 1)  # (B, T, 64)
        aux_output = self.aux_linear(x_aux)  # (B, T, vocab_size + 1)

        # Second Transformer layer
        x = self.transformer_encoder[1](x)

        # Main output: gloss probabilities
        gloss_probs = self.classifier(x)  # (B, T, vocab_size)
        return gloss_probs, aux_output

# Example Usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 10  # Example gloss vocabulary size
    model = TransformerSequenceLearning(
        input_dim=2 * 256,  # 2 hands * 256 features
        model_dim=256,      # Hidden size 256
        num_heads=4,        # 4 heads
        num_layers=2,       # 2 layers
        vocab_size=vocab_size,
        dropout=0.1,
        device=device
    ).to(device)
    x = torch.randn(4, 191, 2, 256).to(device)  # Example input from TemporalEncoding
    gloss_probs, aux_output = model(x)
    print(f"Input shape: {x.shape}")  # [4, 191, 2, 256]
    print(f"Gloss probabilities shape: {gloss_probs.shape}")  # [4, 191, 10]
    print(f"Auxiliary output shape: {aux_output.shape}")  # [4, 191, 11]