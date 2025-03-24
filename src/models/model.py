import os
import tempfile
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.models.decoder import ctc_decoder

from src.models.spatial_encoding.spatial_encoding import SpatialEncoding
from src.models.temporal_encoding.tempconv import TemporalEncoding
from src.models.sequence_learning.transformer import TransformerSequenceLearning
from src.models.alignment.enstim_ctc import EnStimCTC
# from spatial_encoding.spatial_encoding import SpatialEncoding
# from temporal_encoding.tempconv import TemporalEncoding
# from sequence_learning.transformer import TransformerSequenceLearning
# from alignment.enstim_ctc import EnStimCTC


class CSLRModel(nn.Module):
    def __init__(self, spatial_params, temporal_params, transformer_params, enstim_params, label_mapping_path, device='cpu'):
        super(CSLRModel, self).__init__()
        self.device = torch.device(device)
        # Load label-idx mapping from JSON
        with open(label_mapping_path, 'r') as f:
            gloss_to_idx = json.load(f)
        self.idx_to_gloss = {int(idx): gloss for gloss, idx in gloss_to_idx.items()}
        
        # self.idx_to_gloss = idx_to_gloss  # Mapping from indices to gloss names
        
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
            indices = self.enstim_ctc.decode(gloss_probs, input_lengths)
            # Convert indices to gloss strings using self.idx_to_gloss
            glosses = [[self.idx_to_gloss[idx] for idx in seq] for seq in indices]
            return glosses

    def decode_with_lm(self, skeletal, crops, optical_flow, input_lengths, lm_path, beam_size=10, lm_weight=0.5):
        with torch.no_grad():
            spatial_features = self.spatial_encoding(skeletal, crops, optical_flow)
            temporal_features, _ = self.temporal_encoding(spatial_features)
            gloss_probs, _ = self.transformer_sequence_learning(temporal_features)
            log_probs = self.enstim_ctc.decode(gloss_probs, input_lengths, return_log_probs=True)

            # Define tokens (blank, silence, unk, glosses)
            tokens = ["", "|", "<unk>"] + [self.idx_to_gloss[i] for i in range(1, self.enstim_ctc.vocab_size)]

            # Generate lm_dict content (include silence and unk)
            lm_dict_content = ["|", "<unk>"] + [self.idx_to_gloss[i] for i in range(1, self.enstim_ctc.vocab_size)]
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp:
                tmp.write("\n".join(lm_dict_content))
                lm_dict_path = tmp.name

            # Initialize CTC beam decoder
            decoder = ctc_decoder(
                lexicon=None,
                tokens=tokens,
                lm=lm_path,
                lm_dict=lm_dict_path,
                nbest=1,
                beam_size=beam_size,
                beam_size_token=None,
                beam_threshold=50.0,
                lm_weight=lm_weight,
                word_score=0.0,
                unk_score=float('-inf'),
                sil_score=0.0,
                log_add=False,
                blank_token="",
                sil_token="|",
                unk_word="<unk>"
            )

            # Perform beam search decoding
            decoded = decoder(log_probs, input_lengths)

            # Clean up temporary file
            os.unlink(lm_dict_path)

            # Extract and filter the best sequence
            results = []
            for b in range(log_probs.shape[0]):
                best_seq = decoded[b][0][0]  # Tensor of token indices
                # Convert indices to strings using tokens list
                gloss_seq = [tokens[idx.item()] for idx in best_seq]
                # Filter out special tokens
                filtered_seq = [token for token in gloss_seq if token not in ["", "|", "<unk>"]]
                results.append(filtered_seq)
            return results

# Example usage:
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 14  # 13 glosses + 1 blank

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
    
    # Instantiate the model with JSON mapping
    label_mapping_path = "data/label-idx-mapping.json"
    model = CSLRModel(spatial_params, temporal_params, transformer_params, enstim_params, label_mapping_path, device=device).to(device)
    
    # Create dummy input data
    B, T, num_hands = 1, 156, 2
    skeletal = torch.randn(B, T, num_hands, 21, 3).to(device)
    crops = torch.randn(B, T, num_hands, 3, 112, 112).to(device)
    optical_flow = torch.randn(B, T, num_hands, 2, 112, 112).to(device)
    input_lengths = torch.full((B,), T, dtype=torch.long).to(device)
    
    # Greedy decoding (original)
    decoded_greedy = model.decode(skeletal, crops, optical_flow, input_lengths)
    print(f"Greedy Decoded (indices): {decoded_greedy}")
    print(f"Greedy Decoded (glosses): {' '.join([model.idx_to_gloss[idx] for idx in decoded_greedy[0]])}")
    
    # Beam search decoding with LM
    lm_path = "models\checkpoints\kenlm.binary"  # Replace with actual path to your KenLM binary file
    decoded_beam = model.decode_with_lm(skeletal, crops, optical_flow, input_lengths, lm_path=lm_path, beam_size=10)
    print(f"Beam Decoded (glosses): {' '.join(decoded_beam[0])}")