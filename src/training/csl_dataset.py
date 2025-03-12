# csl_dataset.py
import numpy as np
import torch
from pathlib import Path
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

        print(f"File: {file_path}, Crops shape: {crops.shape}")
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

def collate_fn2(batch):
    """Pads sequences in a batch to the maximum length."""
    batch = [item for item in batch if item is not None] # Remove None items
    if not batch:
        raise ValueError("Empty batch")
    skeletal = [item['skeletal'] for item in batch]
    crops = [item['crops'] for item in batch]
    optical_flow = [item['optical_flow'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    # Find max sequence length
    max_T = max(s.shape[0] for s in skeletal)
    max_label_len = max(l.shape[0] for l in labels)  # For variable-length labels
    
    skeletal_padded = torch.stack([
        torch.nn.functional.pad(s, (0, 0, 0, 0, 0, 0, 0, max_T - s.shape[0]), mode='constant', value=0)
        for s in skeletal
    ])
    crops_padded = torch.stack([
        torch.nn.functional.pad(c, (0, 0, 0, 0, 0, 0, 0, max_T - c.shape[0]), mode='constant', value=0)
        for c in crops
    ])
    max_T_flow = max_T - 1
    flow_padded = torch.stack([
        torch.nn.functional.pad(f, (0, 0, 0, 0, 0, 0, 0, max_T_flow - f.shape[0]), mode='constant', value=0)
        for f in optical_flow
    ])
    labels_padded = torch.stack([
        torch.nn.functional.pad(l, (0, max_label_len - l.shape[0]), mode='constant', value=-1)  # -1 as padding token
        for l in labels
    ])
    
    return {
        'skeletal': skeletal_padded,
        'crops': crops_padded,
        'optical_flow': flow_padded,
        'labels': labels_padded
    }

def collate_fn(batch):
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