import torch
import torch.nn as nn
import torch.nn.functional as F

class EnStimCTC(nn.Module):
    """Entropy Stimulated CTC Loss for CSLR."""
    def __init__(self, input_dim, hidden_dim, vocab_size, blank=0, lambda_entropy=0.1):
        """
        Args:
            input_dim (int): Transformer output feature dimension (e.g., 512).
            hidden_dim (int): Hidden dimension for the auxiliary RNN.
            vocab_size (int): Number of unique glosses (excluding blank).
            blank (int): Index of the blank token (default: 0).
            lambda_entropy (float): Weight for entropy regularization (default: 0.1).
        """
        super(EnStimCTC, self).__init__()
        self.blank = blank
        self.lambda_entropy = lambda_entropy
        
        # Unidirectional RNN (GRU) to encode sentence history
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)
        
        # Projection layer: combines Transformer output and RNN hidden states
        self.proj = nn.Linear(input_dim + hidden_dim, vocab_size + 1)
        
        # Log softmax for CTC probabilities
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, x, targets, input_lengths, target_lengths):
        """
        Args:
            x (Tensor): Transformer output, shape (B, T, D).
            targets (Tensor): Target gloss sequences, shape (B, L).
            input_lengths (Tensor): Length of each input sequence, shape (B,).
            target_lengths (Tensor): Length of each target sequence, shape (B,).
        
        Returns:
            Tensor: EnStimCTC loss.
        """
        # Step 1: Encode history with auxiliary RNN
        rnn_output, _ = self.rnn(x)  # (B, T, hidden_dim)
        
        # Step 2: Combine Transformer output with RNN context
        combined = torch.cat([x, rnn_output], dim=2)  # (B, T, D + hidden_dim)
        
        # Step 3: Project to gloss vocabulary (+ blank)
        logits = self.proj(combined)  # (B, T, V+1)
        log_probs = self.log_softmax(logits)  # (B, T, V+1)
        
        # Step 4: Compute CTC loss
        log_probs_ctc = log_probs.transpose(0, 1)  # (T, B, V+1) for CTC
        ctc_loss = F.ctc_loss(
            log_probs_ctc,
            targets,
            input_lengths,
            target_lengths,
            blank=self.blank,
            reduction='mean'
        )
        
        # Step 5: Compute entropy regularization
        probs = torch.exp(log_probs)  # (B, T, V+1)
        entropy = -torch.sum(probs * log_probs, dim=2)  # (B, T)
        entropy_mean = entropy.mean()  # Average over batch and time
        
        # Step 6: Combine losses
        loss = ctc_loss + self.lambda_entropy * entropy_mean
        return loss

    def decode(self, x, input_lengths):
        """
        Greedy decoding for inference.
        
        Args:
            x (Tensor): Transformer output, shape (B, T, D).
            input_lengths (Tensor): Length of each sequence, shape (B,).
        
        Returns:
            list: Predicted gloss sequences.
        """
        with torch.no_grad():
            rnn_output, _ = self.rnn(x)
            combined = torch.cat([x, rnn_output], dim=2)
            logits = self.proj(combined)
            log_probs = self.log_softmax(logits)
            preds = log_probs.argmax(dim=2)  # (B, T)
            decoded = []
            for b in range(x.shape[0]):
                seq = []
                prev = -1
                for t in range(input_lengths[b]):
                    current = preds[b, t].item()
                    # Skip blanks and repeats
                    if current != self.blank and current != prev:
                        seq.append(current)
                    prev = current if current != self.blank else prev
                decoded.append(seq)
            return decoded

# Example Usage
if __name__ == "__main__":
    # Simulated Transformer output: (B, T, D) = (2, 100, 512)
    x = torch.randn(2, 100, 512)
    targets = torch.tensor([[1, 2, 3], [4, 5, 6]])
    input_lengths = torch.tensor([100, 100], dtype=torch.long)
    target_lengths = torch.tensor([3, 3], dtype=torch.long)
    
    # Initialize EnStimCTC
    model = EnStimCTC(input_dim=512, hidden_dim=256, vocab_size=10, blank=0)
    
    # Compute loss
    loss = model(x, targets, input_lengths, target_lengths)
    print(f"EnStimCTC Loss: {loss.item():.4f}")
    
    # Decode predictions
    decoded = model.decode(x, input_lengths)
    print(f"Predicted Gloss Sequences: {decoded}")