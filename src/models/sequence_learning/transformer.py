# transformer.py
import torch
import torch.nn as nn
import math
from performer_pytorch import SelfAttention

# Relative Positional Encoding (learnable)
class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(RelativePositionalEncoding, self).__init__()
        # Learnable positional embeddings
        self.pe = nn.Parameter(torch.randn(max_len, d_model))
    
    def forward(self, x):
        # x shape: (B, T, d_model)
        T = x.size(1)
        return x + self.pe[:T, :].unsqueeze(0)

# Custom Transformer Encoder Layer with Performer Attention and GELU activation
class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, nb_features=512):
        super(CustomTransformerEncoderLayer, self).__init__()
        # Use Performer SelfAttention with adjustable number of random features
        self.self_attn = SelfAttention(
            dim=d_model,
            heads=nhead,
            dim_head=d_model // nhead,
            causal=False,
            dropout=dropout,
            nb_features=nb_features
        )
        # Standard feedforward network using GELU activation
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        nn.init.kaiming_normal_(self.linear1.weight, nonlinearity='linear')
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        nn.init.kaiming_normal_(self.linear2.weight, nonlinearity='linear')
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.norm1(src)
            
        # print("src2 mean:", src2.mean().item(), "std:", src2.std().item(), "max:", src2.abs().max().item())
        if torch.isnan(src2).any() or torch.isinf(src2).any():
            print("NaN or Inf detected in src2 after norm1")

        src2 = torch.clamp(src2, min=-10, max=10)

        variance = src.var(dim=-1, keepdim=True)
        if (variance < 1e-5).any():
            print("Warning: Very small variance in src:", variance.min().item())
            src2 = src2 + 1e-5  # Add a small constant to stabilize

        src2 = src2 / (src2.std(dim=-1, keepdim=True) + 1e-5)
        src2 = src2 / 2.0
        
        attn_output = self.self_attn(src2)
        if torch.isnan(attn_output).any() or torch.isinf(attn_output).any():
            print("NaN or Inf after self_attn")
        
        src = src + self.dropout1(attn_output)
        if torch.isnan(src).any() or torch.isinf(src).any():
            print("NaN or Inf after attention residual")
        
        src2 = self.norm2(src)
        if torch.isnan(src2).any() or torch.isinf(src2).any():
            print("NaN or Inf after norm2")
        
        ff_output = self.linear1(src2)
        if torch.isnan(ff_output).any() or torch.isinf(ff_output).any():
            print("NaN or Inf after linear1")
        
        ff_output = self.activation(ff_output)
        if torch.isnan(ff_output).any() or torch.isinf(ff_output).any():
            print("NaN or Inf after activation")
        
        ff_output = self.dropout(ff_output)
        if torch.isnan(ff_output).any() or torch.isinf(ff_output).any():
            print("NaN or Inf after dropout")
        
        ff_output = self.linear2(ff_output)
        if torch.isnan(ff_output).any() or torch.isinf(ff_output).any():
            print("NaN or Inf after linear2")
        
        src = src + self.dropout2(ff_output)
        if torch.isnan(src).any() or torch.isinf(src).any():
            print("NaN or Inf after ff residual")
        
        return src

# Updated Transformer Sequence Learning with Performer and Relative Positional Encoding,
# now with explicit device handling.
class TransformerSequenceLearning(nn.Module):
    def __init__(self, input_dim, model_dim=256, num_heads=4, num_layers=2,
                 vocab_size=None, dropout=0.1, nb_features=512, device='cpu'):
        super(TransformerSequenceLearning, self).__init__()
        self.device = torch.device(device)
        self.model_dim = model_dim

        # Input projection to model_dim (256)
        self.input_proj = nn.Linear(input_dim, model_dim).to(self.device)
        nn.init.kaiming_normal_(self.input_proj.weight, nonlinearity='linear')
        
        # Use learnable relative positional encoding
        self.pos_encoder = RelativePositionalEncoding(model_dim).to(self.device)
        
        # Transformer encoder with Performer-based layers
        encoder_layers = [CustomTransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,  # Standard FFN size (256*4 = 1024)
            dropout=dropout,
            nb_features=512
        ).to(self.device) for _ in range(num_layers)]
        self.transformer_encoder = nn.ModuleList(encoder_layers).to(self.device)
        
        # Main classification head
        self.classifier = nn.Linear(model_dim, vocab_size).to(self.device)
        nn.init.kaiming_normal_(self.classifier.weight, nonlinearity='linear')
        
        # Auxiliary CTC head (remains unchanged)
        self.aux_conv = nn.Conv1d(model_dim, 64, kernel_size=3, padding=1).to(self.device)
        self.aux_linear = nn.Linear(64, vocab_size).to(self.device)

    def forward(self, x):
        # x: (B, T, num_hands, D), e.g., (B, T, 2, 256)
        x = x.to(self.device)

        x = x / x.std(dim=-1, keepdim=True)  # Normalize to unit variance
        
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("NaN or Inf in input x")
            
        B, T, num_hands, D = x.shape
        # Merge hands into feature dimension: (B, T, num_hands * D)
        x = x.view(B, T, -1)
        x = self.input_proj(x)  # (B, T, model_dim)
        
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("NaN or Inf after input_proj")
            
        x = self.pos_encoder(x)  # Add relative positional encoding

        if torch.isnan(x).any() or torch.isinf(x).any():
            print("NaN or Inf after pos_encoder")
        
        # First Transformer layer with auxiliary CTC head
        x = self.transformer_encoder[0](x)
        
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("NaN or Inf after first transformer layer")
            
        x_aux = x.transpose(1, 2)  # (B, model_dim, T)
        x_aux = self.aux_conv(x_aux)  # (B, 64, T)
        x_aux = x_aux.transpose(1, 2)  # (B, T, 64)
        aux_output = self.aux_linear(x_aux)  # (B, T, vocab_size + 1)

        if torch.isnan(x_aux).any() or torch.isinf(x_aux).any():
            print("NaN or Inf after aux_conv")
        
        # Second Transformer layer
        x = self.transformer_encoder[1](x)

        if torch.isnan(x_aux).any() or torch.isinf(x_aux).any():
            print("NaN or Inf after second transformer layer")
        
        # Main output: gloss probabilities
        gloss_probs = self.classifier(x)  # (B, T, vocab_size)
        return gloss_probs, aux_output

# Example Usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 10  # Example gloss vocabulary size
    model = TransformerSequenceLearning(
        input_dim=2 * 256,  # 2 hands * 256 features
        model_dim=256,
        num_heads=4,
        num_layers=2,
        vocab_size=vocab_size,
        dropout=0.1,
        nb_features=512,
        device=device
    ).to(device)
    x = torch.randn(4, 114, 2, 256).to(device)  # Example input from TemporalEncoding
    gloss_probs, aux_output = model(x)
    print(f"Input shape: {x.shape}")           # [4, 114, 2, 256]
    print(f"Gloss probabilities shape: {gloss_probs.shape}")  # [4, 114, 10]
    print(f"Auxiliary output shape: {aux_output.shape}")      # [4, 114, 11]
