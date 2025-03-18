import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.spatial_encoding.spatial_encoding import SpatialEncoding
from src.models.temporal_encoding.tempconv import TemporalEncoding
from src.models.sequence_learning.transformer import TransformerSequenceLearning
from src.models.alignment.enstim_ctc import EnStimCTC

class CSLRModel(nn.Module):
    def __init__(self, spatial_params, temporal_params, transformer_params, enstim_params, device='cpu'):
        super(CSLRModel, self).__init__()
        self.device = torch.device(device)
        
        # Instantiate each module with the given parameters.
        self.spatial_encoding = SpatialEncoding(**spatial_params, device=device)
        self.temporal_encoding = TemporalEncoding(**temporal_params, device=device)
        self.transformer_sequence_learning = TransformerSequenceLearning(**transformer_params, device=device)
        self.enstim_ctc = EnStimCTC(**enstim_params, device=device)

    def forward(self, skeletal, crops, optical_flow, targets, input_lengths, target_lengths):
        # Step 1: Spatial Encoding produces features of shape (B, T, num_hands, D_spatial)
        spatial_features = self.spatial_encoding(skeletal, crops, optical_flow)
        
        # Step 2: Temporal Encoding fuses spatial features over time;
        # outputs (B, T, num_hands, out_channels) and an auxiliary CTC logits output.
        temporal_features, temporal_aux_logits = self.temporal_encoding(spatial_features)
        
        # Step 3: Transformer Sequence Learning fuses multi-hand temporal features.
        # It expects an input of shape (B, T, num_hands*out_channels) (here: 2*256 = 512)
        gloss_probs, transformer_aux_logits = self.transformer_sequence_learning(temporal_features)
        
        # Step 4: Compute losses.
        # EnStimCTC loss on final gloss probabilities.
        enstim_ctc_loss = self.enstim_ctc(gloss_probs, targets, input_lengths, target_lengths)
        
        # Auxiliary CTC losses (from Temporal and Transformer outputs).
        log_probs_temporal = F.log_softmax(temporal_aux_logits, dim=2).transpose(0, 1)
        log_probs_transformer = F.log_softmax(transformer_aux_logits, dim=2).transpose(0, 1)
        
        ctc_temporal_loss = F.ctc_loss(
            log_probs_temporal, targets, input_lengths, target_lengths,
            blank=self.enstim_ctc.blank, reduction='mean', zero_infinity=True
        )
        ctc_transformer_loss = F.ctc_loss(
            log_probs_transformer, targets, input_lengths, target_lengths,
            blank=self.enstim_ctc.blank, reduction='mean', zero_infinity=True
        )
        
        # Total loss: main EnStimCTC loss plus weighted auxiliary losses.
        total_loss = enstim_ctc_loss + 0.2 * ctc_temporal_loss + 0.2 * ctc_transformer_loss
        return total_loss, enstim_ctc_loss, ctc_temporal_loss, ctc_transformer_loss

    def decode(self, skeletal, crops, optical_flow, input_lengths):
        with torch.no_grad():
            spatial_features = self.spatial_encoding(skeletal, crops, optical_flow)
            temporal_features, _ = self.temporal_encoding(spatial_features)
            gloss_probs, _ = self.transformer_sequence_learning(temporal_features)
            return self.enstim_ctc.decode(gloss_probs, input_lengths)

# Example usage:
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 14  # 13 glosses + 1 blank
    
    # Define module parameter dictionaries.
    spatial_params = {
        "D_spatial": 128  # Output feature dimension from spatial encoding.
    }
    temporal_params = {
        "in_channels": 128,      # Matches D_spatial from the spatial encoder.
        "out_channels": 256,     # Expanded temporal feature dimension.
        "kernel_sizes": [3, 5, 7],
        "dilations": [1, 2, 4],
        "vocab_size": vocab_size
    }
    transformer_params = {
        "input_dim": 2 * 256,    # 2 hands * 256 features = 512.
        "model_dim": 256,
        "num_heads": 4,
        "num_layers": 2,
        "vocab_size": vocab_size,
        "dropout": 0.1
    }
    enstim_params = {
        "vocab_size": vocab_size,
        "context_dim": 256,
        "blank": 0,
        "lambda_entropy": 0.1
    }
    
    # Instantiate the model.
    model = CSLRModel(spatial_params, temporal_params, transformer_params, enstim_params, device=device).to(device)
    
    # Create dummy input data.
    B = 4       # Batch size.
    T = 191     # Sequence length (number of frames).
    num_hands = 2
    skeletal = torch.randn(B, T, num_hands, 21, 3).to(device)
    crops = torch.randn(B, T, num_hands, 3, 112, 112).to(device)
    optical_flow = torch.randn(B, T, num_hands, 2, 112, 112).to(device)
    
    # Dummy target sequences for CTC loss.
    targets = torch.randint(1, vocab_size, (B * 4,), dtype=torch.long).to(device)  # Example: each sample with target length 4.
    input_lengths = torch.full((B,), T, dtype=torch.long).to(device)
    target_lengths = torch.full((B,), 4, dtype=torch.long).to(device)
    
    # Forward pass (training).
    total_loss, enstim_loss, temporal_loss, transformer_loss = model(
        skeletal, crops, optical_flow, targets, input_lengths, target_lengths
    )
    
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"EnStimCTC Loss: {enstim_loss.item():.4f}")
    print(f"Temporal CTC Loss: {temporal_loss.item():.4f}")
    print(f"Transformer CTC Loss: {transformer_loss.item():.4f}")
    
    # Inference (decoding).
    decoded = model.decode(skeletal, crops, optical_flow, input_lengths)
    print(f"Predicted Gloss Sequences: {decoded}")
