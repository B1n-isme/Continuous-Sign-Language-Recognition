import os
import sys
import torch
from torch.utils.data import DataLoader
from jiwer import wer  # For Word Error Rate
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import numpy as np
from tqdm import tqdm
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.model import CSLRModel  # Your CSLR model definition
from src.training.csl_dataset import CSLDataset, collate_fn  # Dataset and collation utilities
from src.utils.label_utils import build_vocab, load_labels  # Vocabulary utilities

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Paths and hyperparameters
    test_dataset_path = "data/datasets/test_dataset.pt"
    labels_csv = "data/labels.csv"
    checkpoint_path = "checkpoints/best_model.pt"
    batch_size = 4

    # Load test dataset
    try:
        test_dataset = torch.load(test_dataset_path, weights_only=False)
    except Exception as e:
        logging.error(f"Failed to load test dataset: {e}")
        raise

    # Build vocabulary
    label_dict = load_labels(labels_csv)
    vocab = build_vocab(label_dict)
    vocab_size = len(vocab)
    logging.info(f"Vocabulary size: {vocab_size}")

    # Inverse vocabulary for decoding
    idx_to_gloss = {idx: gloss for gloss, idx in vocab.items()}

    # Set up DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        drop_last=True
    )
    logging.info("Successfully loaded test dataset and created DataLoader!")

    # Model parameters (adjust to match your training configuration)
    spatial_params = {"D_spatial": 128}
    temporal_params = {
        "in_channels": 128,
        "out_channels": 256,
        "kernel_sizes": [3, 5, 7],
        "dilations": [1, 2, 4],
        "vocab_size": vocab_size
    }
    transformer_params = {
        "input_dim": 2 * 256,  # Adjusted for placeholder
        "model_dim": 256,
        "num_heads": 4,
        "num_layers": 2,
        "vocab_size": vocab_size,
        "dropout": 0.1
    }
    enstim_params = {
        "vocab_size": vocab_size,
        "context_dim": 256,
        "blank": 0,
        "lambda_entropy": 0.1
    }

    # Initialize and load model
    model = CSLRModel(
        spatial_params=spatial_params,
        temporal_params=temporal_params,
        transformer_params=transformer_params,
        enstim_params=enstim_params,
        device=device
    ).to(device)

    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)        
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logging.info(f"Loaded model checkpoint from {checkpoint_path}")
    except Exception as e:
        logging.error(f"Failed to load model checkpoint: {e}")
        raise

    model.eval()

    # Evaluation loop
    all_predictions = []
    all_references = []
    all_pred_indices = []
    all_ref_indices = []
    total_correct = 0
    total_frames = 0

    with torch.no_grad():
        for batch in test_loader:
            skeletal = batch['skeletal'].to(device)
            crops = batch['crops'].to(device)
            optical_flow = batch['optical_flow'].to(device)
            input_lengths = batch['input_lengths'].to(device)
            targets = batch['targets'].to(device)
            target_lengths = batch['target_lengths'].to(device)

            # print(targets)

            # Decode predictions
            pred_sequences = model.decode(skeletal, crops, optical_flow, input_lengths)

            # Split targets into sequences
            ref_sequences = []
            start = 0
            for length in target_lengths:
                end = start + length.item()
                ref_sequences.append(targets[start:end].tolist())
                start = end

            # print(ref_sequences)

            # Store raw indices for debugging
            all_pred_indices.extend(pred_sequences)
            all_ref_indices.extend(ref_sequences)

            # Convert to gloss strings
            pred_glosses = [[idx_to_gloss[idx] for idx in seq] for seq in pred_sequences]
            ref_glosses = [[idx_to_gloss[idx] for idx in seq] for seq in ref_sequences]

            # Collect for metrics
            all_predictions.extend(pred_glosses)
            all_references.extend(ref_glosses)

            # Frame-level accuracy
            for pred, ref in zip(pred_sequences, ref_sequences):
                pred_filtered = [p for p in pred if p != 0]  # Remove blanks
                ref_filtered = [r for r in ref if r != 0]
                correct = sum(p == r for p, r in zip(pred_filtered, ref_filtered))
                total_correct += correct
                total_frames += len(ref_filtered)

    # Compute metrics
    accuracy = total_correct / total_frames if total_frames > 0 else 0.0

    # Join glosses into single strings per sequence for WER
    all_predictions_str = [' '.join(seq) for seq in all_predictions]
    all_references_str = [' '.join(seq) for seq in all_references]

    # print(all_predictions_str)
    # print('\n')
    # print(all_references_str)

    wer_score = wer(all_predictions_str, all_references_str)

    smoothie = SmoothingFunction().method4
    bleu_score = corpus_bleu([[ref] for ref in all_references], all_predictions, smoothing_function=smoothie)

    # Log results
    logging.info("Evaluation Results:")
    logging.info(f"Frame-level Accuracy: {accuracy:.4f}")
    logging.info(f"Word Error Rate (WER): {wer_score:.4f}")
    logging.info(f"BLEU Score: {bleu_score:.4f}")

    # Decode and examine predictions
    logging.info("\nSample Decoded Predictions:")
    for i in range(min(5, len(all_predictions))):
        logging.info(f"Prediction {i+1}: {' '.join(all_predictions[i])}")
        logging.info(f"Reference {i+1}: {' '.join(all_references[i])}")