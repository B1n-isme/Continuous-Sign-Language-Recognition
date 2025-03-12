# split_data.py
import os
import random

def get_unique_sequences(file_dir):
    """Extracts unique sequence identifiers from processed files."""
    sequences = set()
    
    for file in file_dir:
        # Extract filename without path
        filename = os.path.basename(file)
        
        # Split by underscore
        parts = filename.split("_")
        
        if len(parts) < 2:
            continue  # Skip if the filename format is unexpected
        
        # Take first and second parts
        sentence_part = parts[0]
        id_part = parts[1]
        
        # Concatenate both parts
        unique_sequence = f"{sentence_part}_{id_part}"
        
        sequences.add(unique_sequence)

    return list(sequences)

def split_sequences(sequences, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Splits sequences into train/val/test."""
    random.seed(seed)
    random.shuffle(sequences)
    n = len(sequences)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    train_seq = sequences[:train_end]
    val_seq = sequences[train_end:val_end]
    test_seq = sequences[val_end:]
    return train_seq, val_seq, test_seq

def get_file_list(processed_dir, sequences, variants=['original', 'noisy', 'rotated_15', 'rotated_-15', 'warped_slow', 'warped_fast', 'frame_dropped']):
    """Gets all variant files for given sequences."""
    file_list = []
    for seq in sequences:
        for variant in variants:
            file_path = f"{processed_dir}/{seq}_{variant}.npz"
            file_list.append(file_path)
    return file_list