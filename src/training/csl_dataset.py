import numpy as np
import torch
from pathlib import Path
import torch.nn.functional as F
from torch.utils.data import Dataset

class CSLDataset(Dataset):
    def __init__(self, file_list, label_dict, vocab, device='cpu'):
        """
        Args:
            file_list (list): List of file paths to .npz files.
            label_dict (dict): Dictionary mapping file paths to label sequences.
            vocab (dict): Vocabulary mapping words to indices.
            device (str or torch.device): Device to load tensors onto ('cpu', 'cuda', 'mps').
        """
        self.file_list = file_list
        self.label_dict = label_dict
        self.vocab = vocab
        self.device = torch.device(device)
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Get device inside method
        file_path = self.file_list[idx]
        data = np.load(file_path)
        
        # Load modalities
        skeletal_data = data['skeletal_data']  # (T, 2, 21, 3)
        crops = data['crops']                  # (T, 2, 112, 112, 3)
        optical_flow = data['optical_flow']    # (T-1, 2, 112, 112, 2)

        # Ensure exactly 2 hands
        assert crops.shape[1] == 2, f"Expected 2 hands, got {crops.shape[1]} in {file_path}"
        
        # Pad optical flow to T frames
        T = skeletal_data.shape[0]
        zero_flow = np.zeros((1, 2, 112, 112, 2), dtype=optical_flow.dtype)
        optical_flow_padded = np.concatenate([zero_flow, optical_flow], axis=0)  # (T, 2, 112, 112, 2)
        
        # Get targets and convert to indices
        targets = self.label_dict[file_path]
        label_indices = [self.vocab[word] for word in targets]
        
        # Convert to tensors and move to device
        skeletal_tensor = torch.tensor(skeletal_data, dtype=torch.float).to(device)  # (T, 2, 21, 3)
        crops_tensor = torch.tensor(crops, dtype=torch.float).permute(0, 1, 4, 2, 3).to(device)  # (T, 2, 3, 112, 112)
        flow_tensor = torch.tensor(optical_flow_padded, dtype=torch.float).permute(0, 1, 4, 2, 3).to(device)  # (T, 2, 2, 112, 112)
        targets_tensor = torch.tensor(label_indices, dtype=torch.long).to(device)  # (L,)
        
        return {
            'skeletal': skeletal_tensor,
            'crops': crops_tensor,
            'optical_flow': flow_tensor,
            'targets': targets_tensor,
            'input_length': T 
        }

def collate_fn(batch, device='cpu'):
    """Pads sequences to the maximum length and prepares targets for CTC loss.

    Args:
        batch (list): List of samples from CSLDataset.
        device (str or torch.device): Device to load tensors onto ('cpu', 'cuda', 'mps').
    """
    device = torch.device(device)
    
    # Filter out None items
    batch = [item for item in batch if item is not None]
    if not batch:
        raise ValueError("Empty batch")

    # Extract data
    skeletal = [item['skeletal'] for item in batch]          # List of (T_i, 2, 21, 3)
    crops = [item['crops'] for item in batch]                # List of (T_i, 2, 3, 112, 112)
    optical_flow = [item['optical_flow'] for item in batch]  # List of (T_i, 2, 2, 112, 112)
    targets = [item['targets'] for item in batch]              # List of (L_i,)
    input_lengths = [item['input_length'] for item in batch] # List of T_i (Python ints)

    # Determine maximum sequence length
    max_T = max(input_lengths)

    # Pad skeletal
    skeletal_padded = torch.stack([
        F.pad(s, (0, 0, 0, 0, 0, 0, 0, max_T - s.shape[0]), mode='constant', value=0)
        for s in skeletal
    ]).to(device)  # (B, max_T, 2, 21, 3)

    # Pad crops
    crops_padded = torch.stack([
        F.pad(c, (0, 0, 0, 0, 0, 0, 0, 0, 0, max_T - c.shape[0]), mode='constant', value=0)
        for c in crops
    ]).to(device)  # (B, max_T, 2, 3, 112, 112)

    # Pad optical flow
    flow_padded = torch.stack([
        F.pad(f, (0, 0, 0, 0, 0, 0, 0, 0, 0, max_T - f.shape[0]), mode='constant', value=0)
        for f in optical_flow
    ]).to(device)  # (B, max_T, 2, 2, 112, 112)

    # Prepare targets for CTC
    target_lengths = torch.tensor([len(l) for l in targets], dtype=torch.long, device=device)  # (B,)
    targets = torch.cat(targets, dim=0).to(device)  # (sum(L_i),)

    # Convert input_lengths to tensor
    input_lengths = torch.tensor(input_lengths, dtype=torch.long, device=device)  # (B,)

    return {
        'skeletal': skeletal_padded,
        'crops': crops_padded,
        'optical_flow': flow_padded,
        'targets': targets,
        'target_lengths': target_lengths,
        'input_lengths': input_lengths
    }

# Example usage
if __name__ == "__main__":
    # Mock data for testing
    file_list = [Path("data/sample1.npz"), Path("data/sample2.npz")]
    label_dict = {
        Path("data/sample1.npz"): ["hello", "world"],
        Path("data/sample2.npz"): ["sign", "language"]
    }
    vocab = {"hello": 1, "world": 2, "sign": 3, "language": 4, "<blank>": 0}
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Mock npz files would be loaded here; for simplicity, we'll simulate data
    import tempfile
    import os
    for fp in file_list:
        np.savez(fp, skeletal_data=np.random.rand(10, 2, 21, 3),
                 crops=np.random.rand(10, 2, 112, 112, 3),
                 optical_flow=np.random.rand(9, 2, 112, 112, 2))

    dataset = CSLDataset(file_list, label_dict, vocab, device=device)
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=lambda b: collate_fn(b, device=device))

    for batch in dataloader:
        print(f"Skeletal device: {batch['skeletal'].device}")
        print(f"Crops device: {batch['crops'].device}")
        print(f"Optical Flow device: {batch['optical_flow'].device}")
        print(f"Targets device: {batch['targets'].device}")
        print(f"Input lengths device: {batch['input_lengths'].device}")
        print(f"Target lengths device: {batch['target_lengths'].device}")
        break