import os
import sys
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from jiwer import wer  # For Word Error Rate
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import numpy as np
from tqdm import tqdm
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.model import CSLRModel  # Your CSLR model definition
from src.training.csl_dataset import CSLDataset, collate_fn  # Dataset and collation utilities
from src.utils.label_utils import build_vocab, load_labels  # Vocabulary utilities
from src.utils.config_loader import load_config
from src.utils.label_utils import decode_targets

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
            with autocast():
                if use_lm:
                    pred_sequences = model.decode_with_lm(
                        skeletal, crops, optical_flow, input_lengths,
                        lm_path=lm_path, beam_size=beam_size, lm_weight=lm_weight
                    )
                else:
                    pred_sequences = model.decode(skeletal, crops, optical_flow, input_lengths)

            # Split targets into sequences and convert to gloss strings.
            ref_sequences = decode_targets(targets, target_lengths, model.idx_to_gloss)

            # Collect sequences for metrics
            all_predictions.extend(pred_sequences)
            all_references.extend(ref_sequences)

    # Convert sequences to strings: join tokens with a space.
    all_predictions_str = [' '.join(seq) for seq in all_predictions]
    all_references_str = [' '.join(seq) for seq in all_references]

    # Compute corpus-level WER: reference first, prediction second.
    ref_corpus = " ".join(all_references_str)
    pred_corpus = " ".join(all_predictions_str)
    wer_score = wer(ref_corpus, pred_corpus)

    # Compute corpus-level BLEU score using tokenized sentences.
    smoothie = SmoothingFunction().method4
    bleu_score = corpus_bleu(
        [[ref.split()] for ref in all_references_str],
        [pred.split() for pred in all_predictions_str],
        smoothing_function=smoothie
    )

    # Compute Sentence Accuracy: percentage of predictions that exactly match references.
    exact_matches = sum(
        1 for pred, ref in zip(all_predictions_str, all_references_str)
        if pred.strip() == ref.strip()
    )
    accuracy = exact_matches / len(all_references_str) if len(all_references_str) > 0 else 0.0

    # Log results
    method_name = "Beam Search with LM" if use_lm else "Greedy Decoding"
    logging.info(f"\nEvaluation Results ({method_name}):")
    logging.info(f"Word Error Rate (WER): {wer_score:.4f}")
    logging.info(f"BLEU Score: {bleu_score:.4f}")
    logging.info(f"Sentence Accuracy: {accuracy*100:.2f}%")

    # Log a few sample predictions
    logging.info(f"\nSample Decoded Predictions ({method_name}):")
    for i in range(min(5, len(all_predictions))):
        logging.info(f"Prediction {i+1}: {' '.join(all_predictions[i])}")
        logging.info(f"Reference {i+1}: {' '.join(all_references[i])}")

    return wer_score, bleu_score, accuracy

if __name__ == "__main__":
    # Load configuration
    use_lm=False
    model_config = load_config("configs/model_config.yaml")
    lm_path = os.path.join(model_config["checkpoint"]["path"], "kenlm.binary")
    beam_size = model_config["lm_params"]["beam_size"]
    lm_weight = model_config["lm_params"]["weight"]
    data_config = load_config("configs/data_config.yaml")
    test_dataset_path = os.path.join(data_config["paths"]["dataset"], "test_dataset.pt")
    labels_csv = data_config["paths"]["labels"]
    checkpoint_path = model_config["checkpoint"]["path"]
    label_mapping_path = data_config["paths"]["mapping"]
    batch_size = model_config["train_params"]["batch_size"]
    num_workers = model_config["sys_params"]["num_workers"]

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

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
        num_workers=num_workers,
        drop_last=True
    )
    logging.info("Successfully loaded test dataset and created DataLoader!")

    # Model parameters
    spatial_params = model_config["spatial_params"]
    model_config["temporal_params"]["vocab_size"] = vocab_size
    temporal_params = model_config["temporal_params"]
    model_config["transformer_params"]["vocab_size"] = vocab_size
    transformer_params = model_config["transformer_params"]
    model_config["enstim_params"]["vocab_size"] = vocab_size
    enstim_params = model_config["enstim_params"]

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
    greedy_wer, greedy_bleu, greedy_accuracy = evaluate_model(
        model, test_loader, idx_to_gloss, device, use_lm=False
    )

    # Evaluate with beam search decoding if enabled in the static configuration
    if use_lm:
        logging.info("\nStarting evaluation with beam search decoding...")
        lm_wer, lm_bleu, lm_accuracy = evaluate_model(
            model, test_loader, idx_to_gloss, device,
            use_lm=True, lm_path=lm_path, beam_size=beam_size, lm_weight=lm_weight
        )

        # Compare results
        logging.info("\nComparison of Decoding Methods:")
        logging.info(f"Greedy Decoding - WER: {greedy_wer:.4f}, BLEU: {greedy_bleu:.4f}, Accuracy: {greedy_accuracy*100:.2f}%")
        logging.info(f"Beam Search with LM - WER: {lm_wer:.4f}, BLEU: {lm_bleu:.4f}, Accuracy: {lm_accuracy*100:.2f}%")
