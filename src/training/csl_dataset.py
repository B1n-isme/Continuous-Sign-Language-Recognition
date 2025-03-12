# csl_dataset.py
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
        
        # Load modalities (adjust shapes based on your data)
        skeletal_data = data['skeletal_data']  # e.g., (T, 2, 21, 3)
        crops = data['crops']                  # e.g., (T, 2, 112, 112, 3)
        optical_flow = data['optical_flow']    # e.g., (T-1, 2, 112, 112, 2)

        # print(f"File: {file_path}, Crops shape: {crops.shape}")
        assert crops.shape[1] == 2, f"Expected 2 hands, got {crops.shape[1]} in {file_path}"
        
        # Get labels and convert to indices
        labels = self.label_dict[file_path]
        label_indices = [self.vocab[word] for word in labels]
        
        # Convert to tensors
        skeletal_tensor = torch.FloatTensor(skeletal_data)
        crops_tensor = torch.FloatTensor(crops).permute(0, 1, 4, 2, 3)  # (T, 2, 3, 112, 112)
        flow_tensor = torch.FloatTensor(optical_flow).permute(0, 1, 4, 2, 3)  # (T-1, 2, 2, 112, 112)
        labels_tensor = torch.LongTensor(label_indices)
        
        return {
            'skeletal': skeletal_tensor,
            'crops': crops_tensor,
            'optical_flow': flow_tensor,
            'labels': labels_tensor
        }
    
def pad_tensor(t, target_shape):
    # Create a new tensor of zeros with the target shape, same device and dtype as t.
    new_tensor = torch.zeros(target_shape, dtype=t.dtype, device=t.device)
    # Build a tuple of slices for each dimension: [0:s for s in original shape]
    slices = tuple(slice(0, s) for s in t.shape)
    new_tensor[slices] = t
    return new_tensor

def collate_fn(batch):
    """Pads sequences in a batch to the maximum length."""
    batch = [item for item in batch if item is not None] # Remove None items
    if not batch:
        raise ValueError("Empty batch")
    skeletal = [item['skeletal'] for item in batch]
    crops = [item['crops'] for item in batch]
    optical_flow = [item['optical_flow'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    # Determine maximum temporal lengths for skeletal and optical_flow
    max_T = max(s.shape[0] for s in skeletal)
    max_A = max(s.shape[1] for s in skeletal)
    # For optical flow, we need both max temporal length and max "D" dimension
    max_T_flow = max(f.shape[0] for f in optical_flow)
    max_D_flow = max(f.shape[1] for f in optical_flow)
    
    max_label_len = max(l.shape[0] for l in labels)

    # Pad skeletal (assuming skeletal is 4D: [T, ...])
    skeletal_padded = torch.stack([
        pad_tensor(s, (max_T, max_A, s.shape[2], s.shape[3]))
        for s in skeletal
    ])

    # Pad crops (assuming crops is 5D: [T, 2, 3, 112, 112])
    crops_padded = torch.stack([
        F.pad(c, (0, 0, 0, 0, 0, 0, 0, 0, 0, max_T - c.shape[0]), mode='constant', value=0)
        for c in crops
    ])

    # Pad optical flow (5D: [T, D, 2, 112, 112])
    # Padding order: (W_left, W_right, H_left, H_right, C_left, C_right, D_left, D_right, T_left, T_right)
    flow_padded = torch.stack([
        F.pad(
            f,
            (0, 0, 0, 0, 0, 0, 0, max_D_flow - f.shape[1], 0, max_T_flow - f.shape[0]),
            mode='constant',
            value=0
        )
        for f in optical_flow
    ])

    # Pad labels (-1 as the padding token)
    labels_padded = torch.stack([
        F.pad(l, (0, max_label_len - l.shape[0]), mode='constant', value=-1)
        for l in labels
    ])
    
    return {
        'skeletal': skeletal_padded,
        'crops': crops_padded,
        'optical_flow': flow_padded,
        'labels': labels_padded
    }

def collate_fn2(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        raise ValueError("Empty batch")
    
    skeletal = [item['skeletal'] for item in batch]
    crops = [item['crops'] for item in batch]
    optical_flow = [item['optical_flow'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    max_T = max(s.shape[0] for s in skeletal)
    max_label_len = max(l.shape[0] for l in labels)
    
    # Standardize crops to 2 hands
    crops_standardized = []
    for c in crops:
        T, num_hands, C, H, W = c.shape
        if num_hands < 2:
            c_padded = torch.nn.functional.pad(c, (0, 0, 0, 0, 0, 0, 0, 2 - num_hands), mode='constant', value=0)
            crops_standardized.append(c_padded)
        elif num_hands > 2:
            crops_standardized.append(c[:, :2])  # Truncate to 2 hands
        else:
            crops_standardized.append(c)
    
    # Pad frames
    skeletal_padded = torch.stack([
        torch.nn.functional.pad(s, (0, 0, 0, 0, 0, 0, 0, max_T - s.shape[0]), mode='constant', value=0)
        for s in skeletal
    ])
    crops_padded = torch.stack([
        torch.nn.functional.pad(c, (0, 0, 0, 0, 0, 0, 0, max_T - c.shape[0]), mode='constant', value=0)
        for c in crops_standardized
    ])
    max_T_flow = max_T - 1
    flow_padded = torch.stack([
        torch.nn.functional.pad(f, (0, 0, 0, 0, 0, 0, 0, max_T_flow - f.shape[0]), mode='constant', value=0)
        for f in optical_flow
    ])
    labels_padded = torch.stack([
        torch.nn.functional.pad(l, (0, max_label_len - l.shape[0]), mode='constant', value=-1)
        for l in labels
    ])
    
    return {
        'skeletal': skeletal_padded,
        'crops': crops_padded,
        'optical_flow': flow_padded,
        'labels': labels_padded
    }