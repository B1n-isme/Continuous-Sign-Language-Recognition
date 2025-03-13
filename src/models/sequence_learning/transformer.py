import torch
import torch.nn as nn
import math

# Positional Encoding Module (unchanged)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
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
        x = x + self.pe[:x.size(0), :].to(x.device)
        return x

# Improved Transformer Sequence Learning Module
class TransformerSequenceLearning(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, vocab_size, dropout=0.1, device="cpu"):
        super(TransformerSequenceLearning, self).__init__()
        self.device = torch.device(device)  # Store the device
        self.model_dim = model_dim

        # Input projection to reduced model dimension
        self.input_proj = nn.Linear(input_dim, model_dim).to(device)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(model_dim).to(device)

        # Transformer encoder with lightweight settings
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 2,  # Reduced from 4x to 2x
            dropout=dropout,
            activation="relu",
            batch_first=True,
        ).to(device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).to(device)

        # Classification head for gloss probabilities
        self.classifier = nn.Linear(model_dim, vocab_size).to(device)

    def forward(self, x):
        # Ensure tensor is on the correct device
        x = x.to(self.device)
        
        # Input x: (B, T, 2, D)
        B, T, num_hands, D = x.shape
        # Concatenate hand features: (B, T, 2*D)
        x = x.view(B, T, -1)

        # Project to model dimension: (B, T, model_dim)
        x = self.input_proj(x)

        # Add positional encoding (batch_first compatible)
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)

        # Pass through Transformer encoder: (B, T, model_dim)
        x = self.transformer_encoder(x)

        # Output gloss probabilities: (B, T, vocab_size)
        gloss_probs = self.classifier(x)
        return gloss_probs

# Example Usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sample input: (B, T, 2, D) = (4, 191, 2, 256)
    x = torch.randn(4, 191, 2, 256).to(device)

    # Initialize model (vocab_size=100 as an example)
    model = TransformerSequenceLearning(
        input_dim=2 * 256,  # 2 hands * 256 = 512
        model_dim=256,      # Reduced hidden size
        num_heads=4,        # Fewer heads for efficiency
        num_layers=2,       # Lightweight with 2 layers
        vocab_size=10,     # Adjust based on your gloss vocabulary
        dropout=0.1,
        device=device       # Set device
    ).to(device)

    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")  # (4, 191, 2, 256)
    print(f"Output shape: {output.shape}")  # (4, 191, 10)
