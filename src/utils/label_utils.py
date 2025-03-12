# labels_utils.py
import pandas as pd

def load_labels(csv_path):
    """Loads labels.csv into a dictionary: {file_path: [label1, label2, ...]}"""
    df = pd.read_csv(csv_path)
    label_dict = {}
    for _, row in df.iterrows():
        file_path = row['file_path']
        labels = row['labels'].split(',')
        if not labels or labels == ['']:
            print(f"Warning: Empty labels for {file_path}")
            continue
        label_dict[file_path] = labels
    return label_dict

def build_vocab(label_dict):
    """Builds a vocabulary from all unique labels."""
    all_labels = set()
    for labels in label_dict.values():
        all_labels.update(labels)
    vocab = {word: idx for idx, word in enumerate(sorted(all_labels))}
    return vocab