import torch
import torch.nn as nn

# Assuming these modules are defined in separate files within a 'models' directory
from src.models.spatial_encoding.spatial_encoding import SpatialEncoding
from src.models.temporal_encoding.tempconv import TemporalEncoding
from src.models.sequence_learning.transformer import TransformerSequenceLearning
from src.models.alignment.enstim_ctc import EnStimCTC

# from spatial_encoding.spatial_encoding import SpatialEncoding
# from temporal_encoding.tempconv import TemporalEncoding
# from sequence_learning.transformer import TransformerSequenceLearning
# from alignment.enstim_ctc import EnStimCTC

class CSLRModel(nn.Module):
    """
    A unified model for Continuous Sign Language Recognition (CSLR) that integrates:
    - SpatialEncoding: Extracts spatial features from skeletal data, cropped hand images, and optical flow.
    - TemporalEncoding: Captures short-term temporal dependencies using 1D convolutions.
    - TransformerSequenceLearning: Models long-term dependencies and outputs gloss probabilities.
    - EnStimCTC: Applies Entropy Stimulated CTC loss for training or decodes predictions during inference.
    """
    def __init__(self, vocab_size, D_skeletal=64, D_cnn=512, D_flow=512, D_temp=256, model_dim=128,
                 num_heads=4, num_layers=2, context_dim=128, blank=0, lambda_entropy=0.1, dropout=0.1,
                 device='cpu'):
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
            context_dim (int): Dimension for EnStimCTC's causal convolution context.
            blank (int): Index of the blank token in CTC (default: 0).
            lambda_entropy (float): Weight for the entropy term in EnStimCTC loss.
            dropout (float): Dropout rate for the Transformer.
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        super(CSLRModel, self).__init__()
        self.device = torch.device(device)
        D_total = D_skeletal + D_cnn + D_flow

        # Spatial Encoding module
        self.spatial_encoding = SpatialEncoding(D_skeletal, D_cnn, D_flow).to(self.device)

        # Temporal Encoding module (optimized with depthwise separable convolutions)
        self.temporal_encoding = TemporalEncoding(in_channels=D_total, out_channels=D_temp).to(self.device)

        # Lightweight Transformer Sequence Learning module
        self.sequence_learning = TransformerSequenceLearning(
            input_dim=2 * D_temp,  # Concatenated features from both hands
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            vocab_size=vocab_size,
            dropout=dropout,
            device=device
        ).to(self.device)

        # EnStimCTC module with causal convolution
        self.enstim_ctc = EnStimCTC(
            vocab_size=vocab_size,
            context_dim=context_dim,
            blank=blank,
            lambda_entropy=lambda_entropy,
            device=device
        ).to(self.device)

    def forward(self, skeletal, crops, optical_flow, targets=None, input_lengths=None, target_lengths=None):
        """
        Forward pass of the CSLR model.

        Args:
            skeletal (torch.Tensor): Skeletal data of shape (B, T, 2, 21, 3).
            crops (torch.Tensor): Cropped hand images of shape (B, T, 2, 3, 112, 112).
            optical_flow (torch.Tensor): Optical flow of shape (B, T, 2, 2, 112, 112).
            targets (torch.Tensor, optional): Concatenated target gloss sequences of shape (sum(L_i),).
            input_lengths (torch.Tensor, optional): Lengths of input sequences, shape (B,).
            target_lengths (torch.Tensor, optional): Lengths of target sequences, shape (B,).

        Returns:
            torch.Tensor: EnStimCTC loss if targets are provided, or list of decoded gloss sequences if not.
        """

        # Ensure all inputs are on the same device as the model
        skeletal = skeletal.to(self.device)
        crops = crops.to(self.device)
        optical_flow = optical_flow.to(self.device)
        if targets is not None:
            targets = targets.to(self.device)
        if input_lengths is not None:
            input_lengths = input_lengths.to(self.device)
        if target_lengths is not None:
            target_lengths = target_lengths.to(self.device)
            
        # Step 1: Spatial Encoding
        spatial_features = self.spatial_encoding(skeletal, crops, optical_flow)  # (B, T, 2, D_total)

        # Step 2: Temporal Encoding
        temporal_features = self.temporal_encoding(spatial_features)  # (B, T, 2, D_temp)

        # Step 3: Sequence Learning with Transformer
        sequence_output = self.sequence_learning(temporal_features)  # (B, T, vocab_size)

        # Step 4: EnStimCTC for alignment or decoding
        if targets is not None and input_lengths is not None and target_lengths is not None:
            loss = self.enstim_ctc(sequence_output, targets, input_lengths, target_lengths)
            return loss
        else:
            decoded = self.enstim_ctc.decode(sequence_output, input_lengths)
            return decoded

# Example Usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CSLRModel(vocab_size=11, device=device)

    # Define sample inputs
    B, T = 4, 191  # Batch size, time steps
    skeletal = torch.randn(B, T, 2, 21, 3).to(device)
    crops = torch.randn(B, T, 2, 3, 112, 112).to(device)
    optical_flow = torch.randn(B, T, 2, 2, 112, 112).to(device)
    # Concatenated targets matching DataLoader output
    targets = torch.tensor([10, 13, 8, 1, 10, 13, 8, 1, 12, 0, 13, 2, 7, 5, 3, 11], 
                           dtype=torch.long).to(device)  # (sum(L_i),) = (16,)
    input_lengths = torch.tensor([191, 191, 191, 191], dtype=torch.long).to(device)  # (B,) = (4,)
    target_lengths = torch.tensor([4, 4, 4, 4], dtype=torch.long).to(device)  # (B,) = (4,)

    # # Training mode (compute loss)
    # model.train()
    # loss = model(skeletal, crops, optical_flow, targets, input_lengths, target_lengths)
    # print(f"Training Loss: {loss.item():.4f}")

    # # Inference mode (decode sequences)
    # model.eval()
    # with torch.no_grad():
    #     decoded = model(skeletal, crops, optical_flow, input_lengths=input_lengths)
    # print(f"Decoded Sequences: {decoded}")

    # print all shape of input
    print(f"Skeletal shape: {skeletal.shape}")  # (4, 191, 2, 21, 3)
    print(f"Crops shape: {crops.shape}")  # (4, 191, 2, 3, 112, 112)
    print(f"Optical Flow shape: {optical_flow.shape}")  # (4, 191, 2, 2, 112, 112)
    print(f"Targets shape: {targets.shape}")  # (16,)
    print(f"Input Lengths shape: {input_lengths.shape}")  # (4,)
    print(f"Target Lengths shape: {target_lengths.shape}")  # (4,)