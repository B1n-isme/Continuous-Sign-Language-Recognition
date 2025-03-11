# data_loader.py
import os
import torch
from torch.utils.data import DataLoader

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.training.csl_dataset import CSLDataset, collate_fn
from src.training.split_data import get_unique_sequences, split_sequences, get_file_list
from src.utils.label_utils import load_labels, build_vocab

if __name__ == '__main__': 
    # Paths
    processed_dir = 'data/processed'
    labels_csv = 'data/labels.csv'

    # Load labels and build vocabulary
    label_dict = load_labels(labels_csv)
    # print(label_dict)
    vocab = build_vocab(label_dict)
    file_dir = list(label_dict.keys())
    # print(vocab)

    # Get and split unique sequences
    sequences = get_unique_sequences(file_dir)
    train_seq, val_seq, test_seq = split_sequences(sequences)
    # print(sequences)
    # print(f"Train: {len(train_seq)}, Val: {len(val_seq)}, Test: {len(test_seq)}")

    # Get file lists
    train_files = get_file_list(processed_dir, train_seq)
    val_files = get_file_list(processed_dir, val_seq)
    test_files = get_file_list(processed_dir, test_seq)
    # print(test_files)

    # Initialize datasets
    train_dataset = CSLDataset(train_files, label_dict, vocab)
    val_dataset = CSLDataset(val_files, label_dict, vocab)
    test_dataset = CSLDataset(test_files, label_dict, vocab)

    # sample = test_dataset[0]  # Get the first sample
    # print("Sample Keys:", sample.keys())

    # for key, value in sample.items():
    #     print(f"{key}: Type={type(value)}, Shape={value.shape if isinstance(value, torch.Tensor) else 'N/A'}")


    # Create DataLoaders
    batch_size = 4  # Adjust based on your GPU memory
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)

    # Test the train_loader
    for batch in test_loader:
        print(batch.keys())
        print(f"Skeletal: {batch['skeletal'].shape}")
        print(f"Crops: {batch['crops'].shape}")
        print(f"Optical Flow: {batch['optical_flow'].shape}")
        print(f"Labels: {batch['labels'].shape}")
        print(f"Sample labels: {batch['labels'][0]}")
        break