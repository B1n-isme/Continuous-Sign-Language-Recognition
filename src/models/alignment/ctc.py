import torch
import torch.nn as nn
import torch.nn.functional as F

class CTCAlignment(nn.Module):
    """CTC Alignment Module for Continuous Sign Language Recognition."""
    def __init__(self, input_dim, vocab_size, blank=0):
        """
        Args:
            input_dim (int): Dimension of input features (e.g., 512 from sequence model).
            vocab_size (int): Number of unique glosses in the vocabulary.
            blank (int): Index of the blank token (default: 0).
        """
        super(CTCAlignment, self).__init__()
        self.blank = blank
        # Linear layer to project features to vocabulary size + blank token
        self.proj = nn.Linear(input_dim, vocab_size + 1)
        # Log softmax to convert logits to log probabilities
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, x, targets=None, input_lengths=None, target_lengths=None):
        """
        Forward pass for training (CTC loss) or inference (log probabilities).
        
        Args:
            x (Tensor): Input features from sequence model, shape (B, T, D).
            targets (Tensor): Target gloss sequences, shape (B, L) (training only).
            input_lengths (Tensor): Length of each input sequence, shape (B,) (training only).
            target_lengths (Tensor): Length of each target sequence, shape (B,) (training only).
        
        Returns:
            Tensor: CTC loss (training) or log probabilities (inference).
        """
        # Project input features to vocabulary size + blank token
        logits = self.proj(x)  # (B, T, V+1)
        log_probs = self.log_softmax(logits)  # (B, T, V+1)

        if self.training:
            # Transpose to (T, B, V+1) for CTC loss
            log_probs = log_probs.transpose(0, 1)
            # Compute CTC loss
            loss = F.ctc_loss(
                log_probs,
                targets,
                input_lengths,
                target_lengths,
                blank=self.blank,
                reduction='mean'
            )
            return loss
        else:
            # Return log probabilities for inference
            return log_probs

    def greedy_decode(self, log_probs, input_lengths):
        """
        Greedy decoding to convert log probabilities to gloss sequences.
        
        Args:
            log_probs (Tensor): Log probabilities, shape (B, T, V+1).
            input_lengths (Tensor): Length of each sequence, shape (B,).
        
        Returns:
            list: Decoded gloss sequences for each batch item.
        """
        # Get the most likely token at each timestep
        preds = log_probs.argmax(dim=2)  # (B, T)
        decoded = []
        for b in range(preds.size(0)):
            seq = []
            prev = -1  # Previous non-blank token
            # Iterate over the sequence up to its length
            for t in range(input_lengths[b]):
                current = preds[b, t].item()
                # Skip blanks and collapse repeats
                if current != self.blank and current != prev:
                    seq.append(current)
                prev = current if current != self.blank else prev
            decoded.append(seq)
        return decoded

# Example Usage
if __name__ == "__main__":
    # Sample input: (B, T, D) = (4, 191, 512)
    batch_size, seq_length, input_dim = 4, 191, 512
    x = torch.randn(batch_size, seq_length, input_dim)
    
    # Vocabulary: 10 glosses + 1 blank token
    vocab_size = 10
    model = CTCAlignment(input_dim=input_dim, vocab_size=vocab_size, blank=0)
    
    # Sample targets: 4 sequences, each with 5 glosses
    targets = torch.tensor([
        [2, 3, 1, 0, 5],
        [1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4],
        [5, 4, 3, 2, 1]
    ])  # (B, L) = (4, 5)
    input_lengths = torch.full((batch_size,), seq_length, dtype=torch.long)
    target_lengths = torch.full((batch_size,), 5, dtype=torch.long)
    
    # Training: Compute CTC loss
    model.train()
    loss = model(x, targets, input_lengths, target_lengths)
    print(f"CTC Loss: {loss.item():.4f}")
    
    # Inference: Decode predictions
    model.eval()
    with torch.no_grad():
        log_probs = model(x)
        decoded_sequences = model.greedy_decode(log_probs, input_lengths)
    print(f"Decoded Sequences: {decoded_sequences}")