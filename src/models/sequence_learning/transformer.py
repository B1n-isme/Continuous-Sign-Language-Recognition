import torch
import torch.nn as nn
import math


# Positional Encoding Module
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # Shape: (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # Add positional encoding to input
        x = x + self.pe[: x.size(0), :]
        return x


# Transformer Sequence Learning Module
class TransformerSequenceLearning(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerSequenceLearning, self).__init__()
        self.model_dim = model_dim

        # Positional encoding
        self.pos_encoder = PositionalEncoding(model_dim)

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True,  # Set batch_first=True to avoid transposition
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Input and output projection layers
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.output_proj = nn.Linear(
            model_dim, model_dim
        )  # Adjustable based on downstream needs

    def forward(self, x):
        # Input x: (B, T, 2, D)
        B, T, _, D = x.shape
        # Reshape by concatenating hand features: (B, T, 2*D)
        x = x.view(B, T, -1)

        # Project to model dimension: (B, T, model_dim)
        x = self.input_proj(x)

        # Add positional encoding (adapting to batch_first format)
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)

        # Pass through Transformer encoder: (B, T, model_dim)
        x = self.transformer_encoder(x)

        # Project to output dimension: (B, T, model_dim)
        x = self.output_proj(x)

        return x


# Example Usage
if __name__ == "__main__":
    # Sample input from tempconv: (B, T, 2, D) = (4, 191, 2, 256)
    x = torch.randn(4, 191, 2, 256)

    # Initialize the Transformer model
    model = TransformerSequenceLearning(
        input_dim=2 * 256,  # 2 hands * 256 features = 512
        model_dim=512,  # Transformer internal dimension
        num_heads=8,  # Number of attention heads
        num_layers=4,  # Number of Transformer layers
        dropout=0.1,  # Dropout rate
    )

    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")  # (4, 191, 2, 256)
    print(f"Output shape: {output.shape}")  # (4, 191, 512)
