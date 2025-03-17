import torch
import torch.nn as nn
import torch.nn.functional as F

from spatial_encoding.spatial_encoding import SpatialEncoding
from temporal_encoding.tempconv import TemporalEncoding
from sequence_learning.transformer import TransformerSequenceLearning
from alignment.enstim_ctc import EnStimCTC

class CSLRModel(nn.Module):
    def __init__(self, spatial_params, temporal_params, transformer_params, enstim_params, device='cpu'):
        super(CSLRModel, self).__init__()
        self.device = torch.device(device)
        
        # Import your detailed modules here
        self.spatial_encoding = SpatialEncoding(**spatial_params, device=device)
        self.temporal_encoding = TemporalEncoding(**temporal_params, device=device)
        self.transformer_sequence_learning = TransformerSequenceLearning(**transformer_params, device=device)
        self.enstim_ctc = EnStimCTC(**enstim_params, device=device)

    def forward(self, skeletal, crops, optical_flow, targets, input_lengths, target_lengths):
        # Step 1: Spatial encoding
        spatial_features = self.spatial_encoding(skeletal, crops, optical_flow)  # (B, T, num_hands, D_spatial)
        
        # Step 2: Temporal encoding with auxiliary CTC output
        temporal_features, temporal_aux_logits = self.temporal_encoding(spatial_features)
        
        # Step 3: Transformer sequence learning with auxiliary CTC output
        gloss_probs, transformer_aux_logits = self.transformer_sequence_learning(temporal_features)
        
        # Step 4: Compute losses
        # EnStimCTC loss on final output
        enstim_ctc_loss = self.enstim_ctc(gloss_probs, targets, input_lengths, target_lengths)
        
        # Auxiliary CTC losses
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
        
        # Total loss (weighted combination)
        total_loss = enstim_ctc_loss + 0.2 * ctc_temporal_loss + 0.2 * ctc_transformer_loss
        return total_loss, enstim_ctc_loss, ctc_temporal_loss, ctc_transformer_loss

    def decode(self, skeletal, crops, optical_flow, input_lengths):
        with torch.no_grad():
            spatial_features = self.spatial_encoding(skeletal, crops, optical_flow)
            temporal_features, _ = self.temporal_encoding(spatial_features)
            gloss_probs, _ = self.transformer_sequence_learning(temporal_features)
            return self.enstim_ctc.decode(gloss_probs, input_lengths)
        
# Example Usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 14  # 13 glosses + 1 blank
    input_dim = 128  # Example feature dimension
    
    # Dummy data
    x = torch.randn(4, 191, input_dim).to(device)
    targets = torch.tensor([10, 13, 8, 1, 10, 13, 8, 1, 12, 0, 13, 2, 7, 5, 3, 11], dtype=torch.long).to(device)
    input_lengths = torch.tensor([191, 191, 191, 191], dtype=torch.long).to(device)
    target_lengths = torch.tensor([4, 4, 4, 4], dtype=torch.long).to(device)
    
    # Initialize model
    model = CSLRModel(input_dim=input_dim, vocab_size=vocab_size, device=device)
    
    # Training forward pass
    total_loss, enstim_loss, temporal_loss, transformer_loss = model(x, targets, input_lengths, target_lengths)
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"EnStimCTC Loss: {enstim_loss.item():.4f}")
    print(f"Temporal CTC Loss: {temporal_loss.item():.4f}")
    print(f"Transformer CTC Loss: {transformer_loss.item():.4f}")
    
    # Inference
    decoded = model.decode(x, input_lengths)
    print(f"Predicted Gloss Sequences: {decoded}")