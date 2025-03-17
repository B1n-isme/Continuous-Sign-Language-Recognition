import torch
import torch.nn as nn
import torch.nn.functional as F

class EnStimCTC(nn.Module):
    def __init__(self, vocab_size, context_dim, blank=0, lambda_entropy=0.1, device='cpu'):
        super(EnStimCTC, self).__init__()
        self.device = torch.device(device)
        self.blank = blank
        self.lambda_entropy = lambda_entropy
        self.vocab_size = vocab_size
        
        if not (0 <= blank < vocab_size):
            raise ValueError(f"blank ({blank}) must be in range [0, {vocab_size - 1}]")

        self.context_conv = nn.Conv1d(vocab_size, context_dim, kernel_size=3, padding=0, bias=False).to(device)
        self.context_proj = nn.Linear(context_dim, vocab_size).to(device)
        self.combination_weight = nn.Parameter(torch.tensor(0.5, device=device))
        self.log_softmax = nn.LogSoftmax(dim=2).to(device)

    def forward(self, x, targets, input_lengths, target_lengths):
        x = x.to(self.device)
        targets = targets.to(self.device)
        input_lengths = input_lengths.to(self.device)
        target_lengths = target_lengths.to(self.device)

        if targets.max() >= self.vocab_size or targets.min() < 0:
            raise ValueError(f"Target indices must be in [0, {self.vocab_size - 1}]")

        # Encode context with causal convolution
        x_padded = F.pad(x.transpose(1, 2), (2, 0), mode='constant', value=0).transpose(1, 2)
        context = self.context_conv(x_padded.transpose(1, 2))[:, :, :x.size(1)]
        context = context.transpose(1, 2)
        
        # Project context to gloss vocabulary
        context_logits = self.context_proj(context)
        combined_logits = x + torch.clamp(self.combination_weight, 0.0, 1.0) * context_logits
        log_probs = self.log_softmax(combined_logits)
        
        # Optional debugging
        if torch.isnan(log_probs).any():
            print(f"NaN in log_probs! Combined logits range: {combined_logits.min().item()}/{combined_logits.max().item()}")

        # Compute CTC loss
        log_probs_ctc = log_probs.transpose(0, 1)
        ctc_loss = F.ctc_loss(log_probs_ctc, targets, input_lengths, target_lengths,
                              blank=self.blank, reduction='mean', zero_infinity=True)
        
        # Compute entropy regularization
        probs = torch.exp(log_probs.clamp(min=-100, max=0))
        entropy = -torch.sum(probs * log_probs, dim=2)
        entropy_mean = torch.nan_to_num(entropy.mean(), nan=0.0)
        
        # Total loss
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
            combined_logits = x + torch.clamp(self.combination_weight, 0.0, 1.0) * context_logits
            log_probs = self.log_softmax(combined_logits)
            preds = log_probs.argmax(dim=2)
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
    vocab_size = 14  # 13 glosses + 1 blank
    x = torch.randn(4, 191, vocab_size).to(device)  # (B, T, vocab_size) including blank
    targets = torch.tensor([10, 13, 8, 1, 10, 13, 8, 1, 12, 0, 13, 2, 7, 5, 3, 11], 
                           dtype=torch.long).to(device)  # (sum(L_i),), indices < vocab_size
    input_lengths = torch.tensor([191, 191, 191, 191], dtype=torch.long).to(device)
    target_lengths = torch.tensor([4, 4, 4, 4], dtype=torch.long).to(device)
    
    model = EnStimCTC(vocab_size=vocab_size, context_dim=256, blank=0, lambda_entropy=0.1, device=device)
    loss = model(x, targets, input_lengths, target_lengths)
    print(f"EnStimCTC Loss: {loss.item():.4f}")
    decoded = model.decode(x, input_lengths)
    print(f"Predicted Gloss Sequences: {decoded}")