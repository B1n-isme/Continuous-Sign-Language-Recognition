import numpy as np
import torch
from pathlib import Path
import torch.nn.functional as F
from torch.utils.data import Dataset

class CSLDataset(Dataset):
    def __init__(self, file_list, label_dict, vocab):
        self.file_list = file_list
        self.label_dict = label_dict
        self.vocab = vocab
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        data = np.load(file_path)
        
        # Load modalities
        skeletal_data = data['skeletal_data']  # (T, 2, 21, 3)
        crops = data['crops']                  # (T, 2, 112, 112, 3)
        optical_flow = data['optical_flow']    # (T-1, 2, 112, 112, 2)

        # Ensure exactly 2 hands
        assert crops.shape[1] == 2, f"Expected 2 hands, got {crops.shape[1]} in {file_path}"
        
        # Get labels and convert to indices
        labels = self.label_dict[file_path]
        label_indices = [self.vocab[word] for word in labels]
        
        # Convert to tensors
        skeletal_tensor = torch.FloatTensor(skeletal_data)  # (T, 2, 21, 3)
        crops_tensor = torch.FloatTensor(crops).permute(0, 1, 4, 2, 3)  # (T, 2, 3, 112, 112)
        flow_tensor = torch.FloatTensor(optical_flow).permute(0, 1, 4, 2, 3)  # (T-1, 2, 2, 112, 112)
        labels_tensor = torch.LongTensor(label_indices)  # (L,)
        
        # Original sequence length
        T = skeletal_tensor.shape[0]  # Number of frames in the sequence
        
        return {
            'skeletal': skeletal_tensor,
            'crops': crops_tensor,
            'optical_flow': flow_tensor,
            'labels': labels_tensor,
            'input_length': T  # Add original sequence length for CTC
        }
    
def collate_fn(batch):
    """Pads sequences in a batch to the maximum length and returns original lengths."""
    # Filter out None items
    batch = [item for item in batch if item is not None]
    if not batch:
        raise ValueError("Empty batch")

    # Extract data from batch
    skeletal = [item['skeletal'] for item in batch]          # List of (T_i, 2, 21, 3)
    crops = [item['crops'] for item in batch]                # List of (T_i, 2, 3, 112, 112)
    optical_flow = [item['optical_flow'] for item in batch]  # List of (T_i-1, 2, 2, 112, 112)
    labels = [item['labels'] for item in batch]              # List of (L_i,)
    input_lengths = [item['input_length'] for item in batch] # List of original T_i values

    # Determine maximum lengths
    max_T = max(input_lengths)              # Max frames across skeletal and crops
    max_T_flow = max_T - 1                  # Optical flow has one fewer frame
    max_label_len = max(l.shape[0] for l in labels)  # Max label length

    # Pad skeletal (T, 2, 21, 3) -> (max_T, 2, 21, 3)
    skeletal_padded = torch.stack([
        F.pad(s, (0, 0, 0, 0, 0, 0, 0, max_T - s.shape[0]), mode='constant', value=0)
        for s in skeletal
    ])

    # Pad crops (T, 2, 3, 112, 112) -> (max_T, 2, 3, 112, 112)
    crops_padded = torch.stack([
        F.pad(c, (0, 0, 0, 0, 0, 0, 0, 0, 0, max_T - c.shape[0]), mode='constant', value=0)
        for c in crops
    ])

    # Pad optical flow (T-1, 2, 2, 112, 112) -> (max_T-1, 2, 2, 112, 112)
    flow_padded = torch.stack([
        F.pad(f, (0, 0, 0, 0, 0, 0, 0, 0, 0, max_T_flow - f.shape[0]), mode='constant', value=0)
        for f in optical_flow
    ])

    # Pad labels (L,) -> (max_label_len,) with -1 as padding token
    labels_padded = torch.stack([
        F.pad(l, (0, max_label_len - l.shape[0]), mode='constant', value=-1)
        for l in labels
    ])

    # Convert input_lengths to tensor
    input_lengths = torch.tensor(input_lengths, dtype=torch.long)

    return {
        'skeletal': skeletal_padded,       # (B, max_T, 2, 21, 3)
        'crops': crops_padded,             # (B, max_T, 2, 3, 112, 112)
        'optical_flow': flow_padded,       # (B, max_T-1, 2, 2, 112, 112)
        'labels': labels_padded,           # (B, max_label_len)
        'input_lengths': input_lengths     # (B,) - original sequence lengths for CTC
    }