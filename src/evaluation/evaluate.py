import os
import sys
import torch
from torch.utils.data import DataLoader
from jiwer import wer  # For Word Error Rate
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import numpy as np
from tqdm import tqdm
import logging
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.model import CSLRModel  # Your CSLR model definition
from src.training.csl_dataset import CSLDataset, collate_fn  # Dataset and collation utilities
from src.utils.label_utils import build_vocab, load_labels  # Vocabulary utilities

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def evaluate_model(model, test_loader, idx_to_gloss, device, use_lm=False, lm_path=None, beam_size=10, lm_weight=0.5):
    model.eval()
    all_predictions = []
    all_references = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            skeletal = batch['skeletal'].to(device)
            crops = batch['crops'].to(device)
            optical_flow = batch['optical_flow'].to(device)
            input_lengths = batch['input_lengths'].to(device)
            targets = batch['targets'].to(device)
            target_lengths = batch['target_lengths'].to(device)

            # Decode predictions (both return gloss strings)
            if use_lm:
                pred_sequences = model.decode_with_lm(
                    skeletal, crops, optical_flow, input_lengths,
                    lm_path=lm_path, beam_size=beam_size, lm_weight=lm_weight
                )
            else:
                pred_sequences = model.decode(skeletal, crops, optical_flow, input_lengths)

            # Split targets into sequences and convert to glosses
            ref_sequences = []
            start = 0
            for length in target_lengths:
                end = start + length.item()
                seq_indices = targets[start:end].tolist()
                seq_glosses = [idx_to_gloss[idx] for idx in seq_indices]
                ref_sequences.append(seq_glosses)
                start = end

            # Collect for metrics (no further conversion needed for predictions)
            all_predictions.extend(pred_sequences)
            all_references.extend(ref_sequences)

    # Compute metrics
    method_name = "Beam Search with LM" if use_lm else "Greedy Decoding"
    all_predictions_str = [' '.join(seq) for seq in all_predictions]
    all_references_str = [' '.join(seq) for seq in all_references]
    wer_score = wer(all_predictions_str, all_references_str)
    smoothie = SmoothingFunction().method4
    bleu_score = corpus_bleu([[ref] for ref in all_references], all_predictions, smoothing_function=smoothie)

    # Log results
    logging.info(f"\nEvaluation Results ({method_name}):")
    logging.info(f"Word Error Rate (WER): {wer_score:.4f}")
    logging.info(f"BLEU Score: {bleu_score:.4f}")

    # Sample predictions
    logging.info(f"\nSample Decoded Predictions ({method_name}):")
    for i in range(min(5, len(all_predictions))):
        logging.info(f"Prediction {i+1}: {' '.join(all_predictions[i])}")
        logging.info(f"Reference {i+1}: {' '.join(all_references[i])}")

    return wer_score, bleu_score, None  # Accuracy removed

if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate CSLR model with greedy or beam search decoding")
    parser.add_argument('--use_lm', action='store_true', help="Use beam search decoding with language model")
    parser.add_argument('--lm_path', type=str, default="models/checkpoints/kenlm.binary", help="Path to KenLM binary file")
    parser.add_argument('--beam_size', type=int, default=10, help="Beam size for beam search")
    parser.add_argument('--lm_weight', type=float, default=0.5, help="Language model weight for beam search")
    args = parser.parse_args()

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Paths and hyperparameters
    test_dataset_path = "data/datasets/test_dataset.pt"
    labels_csv = "data/labels.csv"
    checkpoint_path = "checkpoints/best_model.pt"
    label_mapping_path = "data/label-idx-mapping.json"
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

    # Inverse vocabulary for decoding references
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

    # Model parameters
    spatial_params = {"D_spatial": 128}
    temporal_params = {
        "in_channels": 128,
        "out_channels": 256,
        "kernel_sizes": [3, 5, 7],
        "dilations": [1, 2, 4],
        "vocab_size": vocab_size
    }
    transformer_params = {
        "input_dim": 2 * 256,
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
        label_mapping_path=label_mapping_path,
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

    # Evaluate with greedy decoding
    logging.info("\nStarting evaluation with greedy decoding...")
    greedy_wer, greedy_bleu, _ = evaluate_model(model, test_loader, idx_to_gloss, device, use_lm=False)

    # Evaluate with beam search decoding if requested
    if args.use_lm:
        logging.info("\nStarting evaluation with beam search decoding...")
        lm_wer, lm_bleu, _ = evaluate_model(
            model, test_loader, idx_to_gloss, device,
            use_lm=True, lm_path=args.lm_path, beam_size=args.beam_size, lm_weight=args.lm_weight
        )

        # Compare results
        logging.info("\nComparison of Decoding Methods:")
        logging.info(f"Greedy Decoding - WER: {greedy_wer:.4f}, BLEU: {greedy_bleu:.4f}")
        logging.info(f"Beam Search with LM - WER: {lm_wer:.4f}, BLEU: {lm_bleu:.4f}")