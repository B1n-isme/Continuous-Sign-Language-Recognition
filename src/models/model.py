import torch
import torch.nn as nn

# Assuming these modules are defined in separate files within a 'models' directory
from spatial_encoding.spatial_encoding import SpatialEncoding
from temporal_encoding.tempconv import TemporalEncoding
from sequence_learning.transformer import TransformerSequenceLearning
from alignment.enstim_ctc import EnStimCTC

class CSLRModel(nn.Module):
    """
    A unified model for Continuous Sign Language Recognition (CSLR) that integrates:
    - SpatialEncoding: Extracts spatial features from skeletal data, cropped hand images, and optical flow.
    - TemporalEncoding: Captures short-term temporal dependencies using 1D convolutions.
    - TransformerSequenceLearning: Models long-term dependencies and outputs gloss probabilities.
    - EnStimCTC: Applies Entropy Stimulated CTC loss for training or decodes predictions during inference.
    """
    def __init__(self, vocab_size, D_skeletal=64, D_cnn=512, D_flow=512, D_temp=256, model_dim=512,
                 num_heads=8, num_layers=4, hidden_dim_rnn=256, blank=0, lambda_entropy=0.1, dropout=0.1):
        """
        Initialize the CSLR model with its submodules.

        Args:
            vocab_size (int): Number of classes including the blank token (e.g., num_glosses + 1).
            D_skeletal (int): Feature dimension for skeletal data.
            D_cnn (int): Feature dimension for CNN-extracted crop features.
            D_flow (int): Feature dimension for optical flow features.
            D_temp (int): Output dimension of temporal encoding.
            model_dim (int): Dimension of the Transformer model.
            num_heads (int): Number of attention heads in the Transformer.
            num_layers (int): Number of Transformer encoder layers.
            hidden_dim_rnn (int): Hidden dimension for the RNN in EnStimCTC.
            blank (int): Index of the blank token in CTC (default: 0).
            lambda_entropy (float): Weight for the entropy term in EnStimCTC loss.
            dropout (float): Dropout rate for the Transformer.
        """
        super(CSLRModel, self).__init__()

        # Total spatial feature dimension
        D_total = D_skeletal + D_cnn + D_flow

        # Spatial Encoding module
        self.spatial_encoding = SpatialEncoding(D_skeletal, D_cnn, D_flow)

        # Temporal Encoding module
        self.temporal_encoding = TemporalEncoding(in_channels=D_total, out_channels=D_temp)

        # Transformer Sequence Learning module (improved to output gloss probabilities)
        self.sequence_learning = TransformerSequenceLearning(
            input_dim=2 * D_temp,  # Concatenated features from both hands
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            vocab_size=vocab_size,  # Includes blank token
            dropout=dropout
        )

        # EnStimCTC module for alignment and loss computation
        self.enstim_ctc = EnStimCTC(
            vocab_size=vocab_size,  # Includes blank token
            hidden_dim=hidden_dim_rnn,
            blank=blank,
            lambda_entropy=lambda_entropy
        )

    def forward(self, skeletal, crops, optical_flow, targets=None, input_lengths=None, target_lengths=None):
        """
        Forward pass of the CSLR model.

        Args:
            skeletal (torch.Tensor): Skeletal data of shape (B, T, 2, 21, 3).
            crops (torch.Tensor): Cropped hand images of shape (B, T, 2, 3, 112, 112).
            optical_flow (torch.Tensor): Optical flow of shape (B, T-1, 2, 2, 112, 112).
            targets (torch.Tensor, optional): Target gloss sequences of shape (B, L).
            input_lengths (torch.Tensor, optional): Lengths of input sequences (B,).
            target_lengths (torch.Tensor, optional): Lengths of target sequences (B,).

        Returns:
            torch.Tensor: EnStimCTC loss during training, or decoded gloss sequences during inference.
        """
        # Step 1: Spatial Encoding
        spatial_features = self.spatial_encoding(skeletal, crops, optical_flow)  # (B, T, 2, D_total)

        # Step 2: Temporal Encoding
        temporal_features = self.temporal_encoding(spatial_features)  # (B, T, 2, D_temp)

        # Step 3: Sequence Learning with Transformer
        sequence_output = self.sequence_learning(temporal_features)  # (B, T, vocab_size)

        # Step 4: EnStimCTC for alignment
        if self.training:
            if targets is None or input_lengths is None or target_lengths is None:
                raise ValueError("Targets, input_lengths, and target_lengths must be provided during training.")
            loss = self.enstim_ctc(sequence_output, targets, input_lengths, target_lengths)
            return loss
        else:
            decoded = self.enstim_ctc.decode(sequence_output, input_lengths)
            return decoded

# Example Usage
if __name__ == "__main__":
    # Define sample input shapes
    B, T, L = 4, 191, 5  # Batch size, time steps, target sequence length
    skeletal = torch.randn(B, T, 2, 21, 3)  # Skeletal data for both hands
    crops = torch.randn(B, T, 2, 3, 112, 112)  # Cropped hand images
    optical_flow = torch.randn(B, T-1, 2, 2, 112, 112)  # Optical flow
    targets = torch.randint(1, 10, (B, L))  # Target gloss indices (1 to 9, assuming blank=0)
    input_lengths = torch.full((B,), T, dtype=torch.long)  # Length of input sequences
    target_lengths = torch.full((B,), L, dtype=torch.long)  # Length of target sequences

    # Initialize the model with vocab_size=10 (9 glosses + 1 blank)
    model = CSLRModel(vocab_size=10, blank=0)

    # Training mode
    model.train()
    loss = model(skeletal, crops, optical_flow, targets, input_lengths, target_lengths)
    print(f"Training Loss: {loss.item():.4f}")

    # Inference mode
    model.eval()
    with torch.no_grad():
        decoded = model(skeletal, crops, optical_flow, input_lengths=input_lengths)
    print(f"Decoded Sequences: {decoded}")