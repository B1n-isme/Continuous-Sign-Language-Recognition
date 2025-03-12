import torch
import torch.nn as nn
import torch.nn.functional as F

class EnStimCTC(nn.Module):
    """Entropy Stimulated CTC Loss for CSLR with Transformer gloss probabilities."""
    def __init__(self, vocab_size, hidden_dim, blank=0, lambda_entropy=0.1):
        """
        Args:
            vocab_size (int): Number of unique glosses (including blank, as output by Transformer).
            hidden_dim (int): Hidden dimension for the auxiliary RNN.
            blank (int): Index of the blank token (default: 0).
            lambda_entropy (float): Weight for entropy regularization (default: 0.1).
        """
        super(EnStimCTC, self).__init__()
        self.blank = blank
        self.lambda_entropy = lambda_entropy
        self.vocab_size = vocab_size
        
        # Unidirectional RNN (GRU) to encode history from Transformer output
        self.rnn = nn.GRU(input_size=vocab_size, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        
        # Projection layer to refine RNN output to gloss probabilities
        self.rnn_proj = nn.Linear(hidden_dim, vocab_size)
        
        # Log softmax for final probabilities
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, x, targets, input_lengths, target_lengths):
        """
        Args:
            x (Tensor): Transformer output logits, shape (B, T, vocab_size).
            targets (Tensor): Target gloss sequences, shape (B, L).
            input_lengths (Tensor): Length of each input sequence, shape (B,).
            target_lengths (Tensor): Length of each target sequence, shape (B,).
        
        Returns:
            Tensor: EnStimCTC loss.
        """
        # Step 1: Encode history with auxiliary RNN
        rnn_output, _ = self.rnn(x)  # (B, T, hidden_dim)
        
        # Step 2: Project RNN output to gloss vocabulary
        rnn_logits = self.rnn_proj(rnn_output)  # (B, T, vocab_size)
        
        # Step 3: Combine Transformer logits with RNN context (StimCTC)
        combined_logits = x + 0.5 * rnn_logits  # Weighted sum of Transformer and RNN logits
        log_probs = self.log_softmax(combined_logits)  # (B, T, vocab_size)
        
        # Step 4: Compute CTC loss
        log_probs_ctc = log_probs.transpose(0, 1)  # (T, B, vocab_size) for CTC
        ctc_loss = F.ctc_loss(
            log_probs_ctc,
            targets,
            input_lengths,
            target_lengths,
            blank=self.blank,
            reduction='mean'
        )
        
        # Step 5: Compute entropy regularization (EnCTC)
        probs = torch.exp(log_probs)  # (B, T, vocab_size)
        entropy = -torch.sum(probs * log_probs, dim=2)  # (B, T)
        entropy_mean = entropy.mean()  # Average over batch and time
        
        # Step 6: Total loss
        loss = ctc_loss + self.lambda_entropy * entropy_mean
        return loss

    def decode(self, x, input_lengths):
        """
        Greedy decoding for inference.
        
        Args:
            x (Tensor): Transformer output logits, shape (B, T, vocab_size).
            input_lengths (Tensor): Length of each sequence, shape (B,).
        
        Returns:
            list: Predicted gloss sequences.
        """
        with torch.no_grad():
            rnn_output, _ = self.rnn(x)  # (B, T, hidden_dim)
            rnn_logits = self.rnn_proj(rnn_output)  # (B, T, vocab_size)
            combined_logits = x + 0.5 * rnn_logits  # Combine with Transformer logits
            log_probs = self.log_softmax(combined_logits)
            preds = log_probs.argmax(dim=2)  # (B, T)
            decoded = []
            for b in range(x.shape[0]):
                seq = []
                prev = -1
                for t in range(input_lengths[b]):
                    current = preds[b, t].item()
                    if current != self.blank and current != prev:
                        seq.append(current)
                    prev = current if current != self.blank else prev
                decoded.append(seq)
            return decoded

# Example Usage
if __name__ == "__main__":
    # Simulated Transformer output: (B, T, vocab_size) = (2, 100, 11) (10 glosses + blank)
    x = torch.randn(4, 191, 11)
    # Target gloss sequences: (B, L) = (4, 5)
    targets = torch.tensor([
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
        [1, 3, 5, 7, 9],
        [4, 6, 8, 2, 1]
    ], dtype=torch.long)
    
    # Input lengths: (B,) = (4,)
    input_lengths = torch.tensor([191, 191, 191, 191], dtype=torch.long)
    
    # Target lengths: (B,) = (4,)
    target_lengths = torch.tensor([5, 5, 5, 5], dtype=torch.long)
    
    # Initialize EnStimCTC
    model = EnStimCTC(
        vocab_size=11,     # 10 glosses + 1 blank
        hidden_dim=256,    # RNN hidden size
        blank=0,           # Blank token index
        lambda_entropy=0.1 # Entropy regularization weight
    )
    
    # Compute loss
    loss = model(x, targets, input_lengths, target_lengths)
    print(f"EnStimCTC Loss: {loss.item():.4f}")
    
    # Decode predictions
    decoded = model.decode(x, input_lengths)
    print(f"Predicted Gloss Sequences: {decoded}")