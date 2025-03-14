import torch
import torch.nn as nn
import torch.nn.functional as F

class EnStimCTC(nn.Module):
    """Improved Entropy Stimulated CTC Loss for CSLR with Transformer gloss probabilities."""
    def __init__(self, vocab_size, context_dim, blank=0, lambda_entropy=0.1, device='cpu'):
        super(EnStimCTC, self).__init__()
        self.device = torch.device(device)
        self.blank = blank
        self.lambda_entropy = lambda_entropy
        self.vocab_size = vocab_size
        
        # Causal 1D convolution for context encoding
        self.context_conv = nn.Conv1d(
            in_channels=vocab_size,
            out_channels=context_dim,
            kernel_size=3,
            padding=0,  # Causal padding handled manually
            bias=False
        ).to(device)
        
        # Projection layer to map context to gloss probabilities
        self.context_proj = nn.Linear(context_dim, vocab_size).to(device)
        
        # Learnable weight for combining logits, initialized to avoid extreme scaling
        self.combination_weight = nn.Parameter(torch.tensor(0.5, device=device))
        
        # Log softmax for probabilities
        self.log_softmax = nn.LogSoftmax(dim=2).to(device)

    def forward(self, x, targets, input_lengths, target_lengths):
        """
        Args:
            x (Tensor): Transformer output logits, shape (B, T, vocab_size).
            targets (Tensor): Concatenated target gloss sequences, shape (sum(L_i),).
            input_lengths (Tensor): Length of each input sequence, shape (B,).
            target_lengths (Tensor): Length of each target sequence, shape (B,).
        
        Returns:
            Tensor: EnStimCTC loss.
        """
        x = x.to(self.device)
        targets = targets.to(self.device)
        input_lengths = input_lengths.to(self.device)
        target_lengths = target_lengths.to(self.device)

        # Step 1: Encode context with causal convolution
        x_padded = F.pad(x.transpose(1, 2), (2, 0), mode='constant', value=0).transpose(1, 2)  # (B, T+2, vocab_size)
        context = self.context_conv(x_padded.transpose(1, 2))[:, :, :x.size(1)]  # (B, context_dim, T)
        context = context.transpose(1, 2)  # (B, T, context_dim)
        
        # Step 2: Project context to gloss vocabulary
        context_logits = self.context_proj(context)  # (B, T, vocab_size)
        
        # Step 3: Combine Transformer and context logits with clamping to prevent explosion
        combined_logits = x + torch.clamp(self.combination_weight, min=0.0, max=1.0) * context_logits
        log_probs = self.log_softmax(combined_logits)  # (B, T, vocab_size)
        
        # Debugging: Check for NaN in log_probs
        if torch.isnan(log_probs).any():
            print("NaN detected in log_probs!")
            print(f"Combined logits min/max: {combined_logits.min().item()}/{combined_logits.max().item()}")
        
        # Step 4: Compute CTC loss
        log_probs_ctc = log_probs.transpose(0, 1)  # (T, B, vocab_size)
        ctc_loss = F.ctc_loss(
            log_probs_ctc,
            targets,
            input_lengths,
            target_lengths,
            blank=self.blank,
            reduction='mean',
            zero_infinity=True  # Handle infinities gracefully
        )
        
        # Debugging: Check CTC loss
        if torch.isnan(ctc_loss):
            print("NaN detected in CTC loss!")
            print(f"log_probs_ctc min/max: {log_probs_ctc.min().item()}/{log_probs_ctc.max().item()}")
        
        # Step 5: Compute entropy regularization (encourage peakiness) with numerical stability
        probs = torch.exp(log_probs.clamp(min=-100, max=0))  # Avoid extreme values
        entropy = -torch.sum(probs * log_probs, dim=2)  # (B, T)
        entropy_mean = torch.nan_to_num(entropy.mean(), nan=0.0)  # Replace NaN with 0
        
        # Debugging: Check entropy
        if torch.isnan(entropy_mean):
            print("NaN detected in entropy_mean!")
            print(f"Entropy min/max: {entropy.min().item()}/{entropy.max().item()}")
        
        # Step 6: Total loss with negative entropy term
        loss = ctc_loss - self.lambda_entropy * entropy_mean
        return loss

    def decode(self, x, input_lengths):
        x = x.to(self.device)
        input_lengths = input_lengths.to(self.device)

        with torch.no_grad():
            x_padded = F.pad(x.transpose(1, 2), (2, 0), mode='constant', value=0).transpose(1, 2)
            context = self.context_conv(x_padded.transpose(1, 2))[:, :, :x.size(1)]
            context = context.transpose(1, 2)
            context_logits = self.context_proj(context)
            combined_logits = x + torch.clamp(self.combination_weight, min=0.0, max=1.0) * context_logits
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(4, 191, 11).to(device)  # (B, T, vocab_size)
    targets = torch.tensor([10, 13, 8, 1, 10, 13, 8, 1, 12, 0, 13, 2, 7, 5, 3, 11], 
                           dtype=torch.long).to(device)
    input_lengths = torch.tensor([191, 191, 191, 191], dtype=torch.long).to(device)
    target_lengths = torch.tensor([4, 4, 4, 4], dtype=torch.long).to(device)
    
    model = EnStimCTC(vocab_size=11, context_dim=256, blank=0, lambda_entropy=0.1, device=device)
    loss = model(x, targets, input_lengths, target_lengths)
    print(f"EnStimCTC Loss: {loss.item():.4f}")
    decoded = model.decode(x, input_lengths)
    print(f"Predicted Gloss Sequences: {decoded}")