# data_loader.py
import os
import torch
from torch.utils.data import DataLoader

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.training.csl_dataset import CSLDataset, collate_fn
from src.training.split_data import get_unique_sequences, split_sequences, get_file_list
from src.utils.label_utils import load_labels, build_vocab

if __name__ == "__main__":
    # Paths
    processed_dir = "data/processed"
    labels_csv = "data/labels.csv"
    datasets_dir = "data/datasets"

    # Load labels and build vocabulary
    label_dict = load_labels(labels_csv)
    # print(label_dict)
    vocab = build_vocab(label_dict)
    # print(vocab)
    file_dir = list(label_dict.keys())

    # Get and split unique sequences
    sequences = get_unique_sequences(file_dir)
    # print(sequences)
    train_seq, val_seq, test_seq = split_sequences(sequences)

    # Get file lists
    train_files = [
        f for f in get_file_list(processed_dir, train_seq) if os.path.exists(f)
    ]
    val_files = [f for f in get_file_list(processed_dir, val_seq) if os.path.exists(f)]
    test_files = [
        f for f in get_file_list(processed_dir, test_seq) if os.path.exists(f)
    ]

    # Initialize datasets
    train_dataset = CSLDataset(train_files, label_dict, vocab)
    val_dataset = CSLDataset(val_files, label_dict, vocab)
    test_dataset = CSLDataset(test_files, label_dict, vocab)

    # for i in range(len(train_files)):
    #     print(train_dataset[i]["labels"])

    # Save datasets and metadata
    print("Saving datasets to", datasets_dir)

    # Save datasets
    torch.save(train_dataset, os.path.join(datasets_dir, "train_dataset.pt"))
    torch.save(val_dataset, os.path.join(datasets_dir, "val_dataset.pt"))
    torch.save(test_dataset, os.path.join(datasets_dir, "test_dataset.pt"))

    # Save metadata
    metadata = {
        "vocab": vocab,
        "label_dict": label_dict,
        "train_files": train_files,
        "val_files": val_files,
        "test_files": test_files,
    }
    torch.save(metadata, os.path.join(datasets_dir, "metadata.pt"))

    # Create DataLoaders
    batch_size = 4  # Adjust based on your GPU memory
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    # Test loading saved datasets
    print("Testing loading datasets...")
    loaded_train_dataset = torch.load(
        os.path.join(datasets_dir, "train_dataset.pt"), weights_only=False
    )
    loaded_metadata = torch.load(os.path.join(datasets_dir, "metadata.pt"))

    # Create a dataloader from loaded dataset
    loaded_train_loader = DataLoader(
        loaded_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
    )

    print("Successfully loaded datasets!")

    # Test the train_loader
    for batch in loaded_train_loader:
        print(batch.keys())
        print(f"Skeletal: {batch['skeletal'].shape}")
        print(f"Crops: {batch['crops'].shape}")
        print(f"Optical Flow: {batch['optical_flow'].shape}")
        print(f"Labels: {batch['labels'].shape}")
        print(f"Sample labels: {batch['labels'][0]}")
