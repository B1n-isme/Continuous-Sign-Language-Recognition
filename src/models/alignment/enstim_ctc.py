# Create a Gaussian Kernel:
# Define a 1D Gaussian kernel based on a specified kernel_size and sigma. This kernel will determine how much the confidence is spread to neighboring frames.
# Apply Smoothing:
# Use a 1D convolution with the Gaussian kernel on the combined logits, ensuring each label (vocabulary class) is smoothed independently along the time axis.
# Integrate into EnStimCTC:
# Add this smoothing step in both the forward method (for training) and the decode method (for inference) to maintain consistency.
# Adjustable Parameters:
# Make kernel_size and sigma configurable hyperparameters to control the extent of smoothing.

import torch
import torch.nn as nn
import torch.nn.functional as F

class RadialEnStimCTC(nn.Module):
    def __init__(self, vocab_size, context_dim, blank=0, lambda_entropy=0.1, kernel_size=5, sigma=1.0, device='cpu'):
        super(RadialEnStimCTC, self).__init__()
        self.device = torch.device(device)
        self.blank = blank
        self.lambda_entropy = lambda_entropy
        self.vocab_size = vocab_size
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.padding = (kernel_size - 1) // 2  # Preserves input length
        
        if not (0 <= blank < vocab_size):
            raise ValueError(f"blank ({blank}) must be in range [0, {vocab_size - 1}]")

        # Existing context convolution and projection
        self.context_conv = nn.Conv1d(vocab_size, context_dim, kernel_size=3, padding=0, bias=False).to(device)
        self.context_proj = nn.Linear(context_dim, vocab_size).to(device)
        self.combination_weight = nn.Parameter(torch.tensor(0.5, device=device))
        self.log_softmax = nn.LogSoftmax(dim=2).to(device)
        
        # Gaussian kernel for radial smoothing, registered as a buffer (non-learnable)
        self.register_buffer('gaussian_kernel', self.create_gaussian_kernel(kernel_size, sigma))

    def create_gaussian_kernel(self, kernel_size, sigma):
        """Creates a 1D Gaussian kernel for smoothing."""
        x = torch.arange(kernel_size, dtype=torch.float32, device=self.device)
        x = x - (kernel_size - 1) / 2  # Center at 0
        kernel = torch.exp(-x**2 / (2 * sigma**2))
        kernel /= kernel.sum()  # Normalize
        kernel = kernel.view(1, 1, -1).repeat(self.vocab_size, 1, 1)  # Shape: (vocab_size, 1, kernel_size)
        return kernel

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
        
        # Project context to vocabulary size
        context_logits = self.context_proj(context)
        combined_logits = x + torch.clamp(self.combination_weight, 0.0, 1.0) * context_logits
        
        # Apply radial smoothing
        combined_logits = combined_logits.transpose(1, 2)  # (B, vocab_size, T)
        smoothed_logits = F.conv1d(combined_logits, self.gaussian_kernel, groups=self.vocab_size, padding=self.padding)
        smoothed_logits = smoothed_logits.transpose(1, 2)  # (B, T, vocab_size)
        
        log_probs = self.log_softmax(smoothed_logits)
        
        # Optional debugging
        if torch.isnan(log_probs).any():
            print(f"NaN in log_probs! Smoothed logits range: {smoothed_logits.min().item()}/{smoothed_logits.max().item()}")

        # Compute CTC loss
        log_probs_ctc = log_probs.transpose(0, 1)  # (T, B, vocab_size)
        ctc_loss = F.ctc_loss(log_probs_ctc, targets, input_lengths, target_lengths,
                              blank=self.blank, reduction='mean', zero_infinity=True)
        
        # Compute entropy regularization
        probs = torch.exp(log_probs.clamp(min=-100, max=0))
        entropy = -torch.sum(probs * log_probs, dim=2)
        entropy_mean = torch.nan_to_num(entropy.mean(), nan=0.0)
        
        # Total loss
        loss = ctc_loss - self.lambda_entropy * entropy_mean
        return loss

    def decode(self, x, input_lengths, return_log_probs=False):
        x = x.to(self.device)
        input_lengths = input_lengths.to(self.device)
        with torch.no_grad():
            x_padded = F.pad(x.transpose(1, 2), (2, 0), mode='constant', value=0).transpose(1, 2)
            context = self.context_conv(x_padded.transpose(1, 2))[:, :, :x.size(1)]
            context = context.transpose(1, 2)
            context_logits = self.context_proj(context)
            combined_logits = x + torch.clamp(self.combination_weight, 0.0, 1.0) * context_logits
            
            # Apply radial smoothing
            combined_logits = combined_logits.transpose(1, 2)  # (B, vocab_size, T)
            smoothed_logits = F.conv1d(combined_logits, self.gaussian_kernel, groups=self.vocab_size, padding=self.padding)
            smoothed_logits = smoothed_logits.transpose(1, 2)  # (B, T, vocab_size)
            
            log_probs = self.log_softmax(smoothed_logits)
            if return_log_probs:
                return log_probs  # Shape: (B, T, vocab_size)
            else:
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
    
    model = RadialEnStimCTC(vocab_size=vocab_size, context_dim=256, blank=0, lambda_entropy=0.1, 
                            kernel_size=5, sigma=1.0, device=device)
    loss = model(x, targets, input_lengths, target_lengths)
    print(f"RadialEnStimCTC Loss: {loss.item():.4f}")
    decoded = model.decode(x, input_lengths)
    print(f"Predicted Gloss Sequences: {decoded}")