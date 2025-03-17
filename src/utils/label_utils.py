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
    """
    Builds a vocabulary from all unique labels, including the blank token at index 0.

    Args:
        label_dict (dict): Dictionary mapping file paths to lists of labels.

    Returns:
        dict: Vocabulary mapping words to indices, with '<blank>' as 0.
    """
    all_labels = set()
    for labels in label_dict.values():
        all_labels.update(labels)
    
    # Start with blank token at index 0
    vocab = {"<blank>": 0}
    
    # Assign indices to glosses starting from 1
    for idx, word in enumerate(sorted(all_labels), start=1):
        vocab[word] = idx

    return vocab