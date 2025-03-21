import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.pipelines import CTCDecoder

from src.models.spatial_encoding.spatial_encoding import SpatialEncoding
from src.models.temporal_encoding.tempconv import TemporalEncoding
from src.models.sequence_learning.transformer import TransformerSequenceLearning
from src.models.alignment.enstim_ctc import EnStimCTC
# from spatial_encoding.spatial_encoding import SpatialEncoding
# from temporal_encoding.tempconv import TemporalEncoding
# from sequence_learning.transformer import TransformerSequenceLearning
# from alignment.enstim_ctc import EnStimCTC

class CSLRModel(nn.Module):
    def __init__(self, spatial_params, temporal_params, transformer_params, enstim_params, idx_to_gloss, device='cpu'):
        super(CSLRModel, self).__init__()
        self.device = torch.device(device)
        self.idx_to_gloss = idx_to_gloss  # Mapping from indices to gloss names
        
        self.spatial_encoding = SpatialEncoding(**spatial_params, device=device)
        self.temporal_encoding = TemporalEncoding(**temporal_params, device=device)
        self.transformer_sequence_learning = TransformerSequenceLearning(**transformer_params, device=device)
        self.enstim_ctc = EnStimCTC(**enstim_params, device=device)

    def forward(self, skeletal, crops, optical_flow, targets, input_lengths, target_lengths):
        # Step 1: Spatial Encoding produces features of shape (B, T, num_hands, D_spatial)
        spatial_features = self.spatial_encoding(skeletal, crops, optical_flow)
        
        # Step 2: Temporal Encoding fuses spatial features over time
        temporal_features, temporal_aux_logits = self.temporal_encoding(spatial_features)
        
        # Step 3: Transformer Sequence Learning fuses multi-hand temporal features
        gloss_probs, transformer_aux_logits = self.transformer_sequence_learning(temporal_features)
        
        # Step 4: Compute losses
        enstim_ctc_loss = self.enstim_ctc(gloss_probs, targets, input_lengths, target_lengths)
        
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
        
        total_loss = enstim_ctc_loss + 0.2 * ctc_temporal_loss + 0.2 * ctc_transformer_loss
        return total_loss, enstim_ctc_loss, ctc_temporal_loss, ctc_transformer_loss

    def decode(self, skeletal, crops, optical_flow, input_lengths):
        with torch.no_grad():
            spatial_features = self.spatial_encoding(skeletal, crops, optical_flow)
            temporal_features, _ = self.temporal_encoding(spatial_features)
            gloss_probs, _ = self.transformer_sequence_learning(temporal_features)
            return self.enstim_ctc.decode(gloss_probs, input_lengths)

    def decode_with_lm(self, skeletal, crops, optical_flow, input_lengths, lm_path, beam_size=10, lm_weight=0.5):
        """
        Decode using beam search with an external n-gram language model via torchaudio.
        
        Args:
            skeletal (Tensor): Skeletal data (B, T, num_hands, 21, 3)
            crops (Tensor): Hand crop images (B, T, num_hands, 3, H, W)
            optical_flow (Tensor): Optical flow (B, T, num_hands, 2, H, W)
            input_lengths (Tensor): Lengths of input sequences (B,)
            lm_path (str): Path to the KenLM binary file (e.g., model.klm)
            beam_size (int): Number of beams for beam search
            lm_weight (float): Weight for LM score
        
        Returns:
            list: Decoded gloss sequences as lists of strings, one per batch item
        """
        with torch.no_grad():
            spatial_features = self.spatial_encoding(skeletal, crops, optical_flow)
            temporal_features, _ = self.temporal_encoding(spatial_features)
            gloss_probs, _ = self.transformer_sequence_learning(temporal_features)
            log_probs = self.enstim_ctc.decode(gloss_probs, input_lengths, return_log_probs=True)
            # log_probs shape: (B, T, vocab_size)

            # Define tokens (gloss names, with "" for blank)
            tokens = [""] + [self.idx_to_gloss[i] for i in range(1, self.enstim_ctc.vocab_size)]

            # Initialize CTC beam decoder
            decoder = CTCDecoder(
                lexicon=None,  # No lexicon (word-to-token mapping) needed for glosses
                tokens=tokens,
                lm=lm_path,
                blank_token="",
                sil_token=None,  # No silence token needed
                unk_word=None,   # No unknown word handling
                nbest=1,         # Return only the best sequence
                beam_size=beam_size,
                lm_weight=lm_weight,
                word_weight=0.0,  # No extra word insertion penalty
                lex_weight=0.0    # No lexicon weight (not using lexicon)
            )

            # Perform beam search decoding
            decoded = decoder(log_probs, input_lengths)

            # Extract the best sequence for each batch item
            results = []
            for b in range(log_probs.shape[0]):
                best_seq = decoded[b][0][0]  # (nbest, (tokens, score)) -> tokens of best hypothesis
                results.append(best_seq)
            return results

# Example usage:
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 14  # 13 glosses + 1 blank
    
    # Define idx_to_gloss mapping
    idx_to_gloss = {
        0: "",  # Blank token
        1: "HELLO",
        2: "I",
        3: "HAVE",
        4: "GOOD",
        5: "LUNCH",
        6: "YOU",
        7: "AND",
        8: "WHAT",
        9: "ARE",
        10: "DO",
        11: "PLEASE",
        12: "THANK",
        13: "BYE"
    }

    # Define module parameter dictionaries
    spatial_params = {"D_spatial": 128}
    temporal_params = {
        "in_channels": 128,
        "out_channels": 256,
        "kernel_sizes": [3, 5, 7],
        "dilations": [1, 2, 4],
        "vocab_size": vocab_size
    }
    transformer_params = {
        "input_dim": 2 * 256,
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
    
    # Instantiate the model
    model = CSLRModel(spatial_params, temporal_params, transformer_params, enstim_params, idx_to_gloss, device=device).to(device)
    
    # Create dummy input data
    B, T, num_hands = 1, 156, 2
    skeletal = torch.randn(B, T, num_hands, 21, 3).to(device)
    crops = torch.randn(B, T, num_hands, 3, 112, 112).to(device)
    optical_flow = torch.randn(B, T, num_hands, 2, 112, 112).to(device)
    input_lengths = torch.full((B,), T, dtype=torch.long).to(device)
    
    # Greedy decoding (original)
    decoded_greedy = model.decode(skeletal, crops, optical_flow, input_lengths)
    print(f"Greedy Decoded (indices): {decoded_greedy}")
    print(f"Greedy Decoded (glosses): {' '.join([idx_to_gloss[idx] for idx in decoded_greedy[0]])}")
    
    # Beam search decoding with LM
    lm_path = "path/to/model.klm"  # Replace with actual path to your KenLM binary file
    decoded_beam = model.decode_with_lm(skeletal, crops, optical_flow, input_lengths, lm_path=lm_path, beam_size=10)
    print(f"Beam Decoded (glosses): {' '.join(decoded_beam[0])}")