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

def collate_fn(batch):
    """Pads sequences in a batch to the maximum length."""
    skeletal = [item['skeletal'] for item in batch]
    crops = [item['crops'] for item in batch]
    optical_flow = [item['optical_flow'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    # Find max sequence length
    max_T = max(s.shape[0] for s in skeletal)
    
    # Pad skeletal and crops to max_T
    skeletal_padded = torch.stack([
        torch.nn.functional.pad(s, (0, 0, 0, 0, 0, 0, 0, max_T - s.shape[0]), mode='constant', value=0)
        for s in skeletal
    ])
    crops_padded = torch.stack([
        torch.nn.functional.pad(c, (0, 0, 0, 0, 0, 0, 0, max_T - c.shape[0]), mode='constant', value=0)
        for c in crops
    ])
    
    # Pad optical flow to max_T - 1
    max_T_flow = max_T - 1
    flow_padded = torch.stack([
        torch.nn.functional.pad(f, (0, 0, 0, 0, 0, 0, 0, max_T_flow - f.shape[0]), mode='constant', value=0)
        for f in optical_flow
    ])
    
    # Stack labels (assuming fixed label length per sequence; pad if variable)
    labels_stacked = torch.stack(labels)
    
    return {
        'skeletal': skeletal_padded,
        'crops': crops_padded,
        'optical_flow': flow_padded,
        'labels': labels_stacked
    }