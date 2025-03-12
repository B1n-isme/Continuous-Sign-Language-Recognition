import os
import sys
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.model import CSLRModel  # Your CSLR model definition
from src.training.csl_dataset import CSLDataset, collate_fn  # Dataset and collation utilities
from src.utils.label_utils import build_vocab, load_labels  # Vocabulary utilities

from jiwer import wer  # For Word Error Rate
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    test_dataset_path = "data/datasets/test_dataset.pt"
    labels_csv = "data/labels.csv"
    model_path = "checkpoints/best_model.pth"

    # Load test dataset and vocabulary
    test_dataset = torch.load(test_dataset_path, weights_only=False)
    label_dict = load_labels(labels_csv)
    vocab = build_vocab(label_dict)
    vocab_size = len(vocab)

    # Initialize and load model
    model = CSLRModel(vocab_size=vocab_size, blank=0).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Set up DataLoader
    batch_size = 4
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, 
        num_workers=2,
        drop_last=True
    )

    # Metric functions
    def compute_accuracy(preds, targets):
        correct = sum([1 for p, t in zip(preds, targets) if p == t])
        return correct / len(targets)

    def compute_wer(preds, targets):
        return wer(targets, preds)

    def compute_bleu(pred, target):
        return sentence_bleu([target], pred)

    ### 1. Loss Computation Pass (using train mode)
    test_loss = 0.0
    num_batches = 0

    # Use torch.no_grad() to disable gradients during evaluation
    with torch.no_grad():
        # Force training mode so that the forward method returns loss
        model.train()
        for batch in tqdm(test_loader, desc="Evaluating Loss"):
            skeletal = batch['skeletal'].to(device)
            crops = batch['crops'].to(device)
            optical_flow = batch['optical_flow'].to(device)
            targets = batch['labels'].to(device)
            input_lengths = torch.full((skeletal.size(0),), skeletal.size(1), dtype=torch.long).to(device)
            target_lengths = torch.sum(targets != -1, dim=1).to(device)

            # Forward pass returns loss when targets are provided
            loss = model(skeletal, crops, optical_flow, targets, input_lengths, target_lengths)

            # Handle both list and tensor cases for loss
            if isinstance(loss, list):
                loss_tensor = torch.stack(tuple(loss)).sum()
            elif isinstance(loss, torch.Tensor):
                loss_tensor = loss.sum() if loss.dim() > 0 else loss
            else:
                raise TypeError("Loss is neither a list nor a tensor.")

            test_loss += loss_tensor.item()
            num_batches += 1

    avg_loss = test_loss / num_batches
    print(f"Test Loss: {avg_loss:.4f}")

    ### 2. Prediction Pass (using eval mode)
    all_preds = []
    all_targets = []
    # Prepare inverse vocabulary for converting indices to strings
    inv_vocab = {v: k for k, v in vocab.items()}

    # Switch model to evaluation mode for decoding predictions
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Predictions"):
            skeletal = batch['skeletal'].to(device)
            crops = batch['crops'].to(device)
            optical_flow = batch['optical_flow'].to(device)
            targets = batch['labels'].cpu().numpy()  # For metric conversion
            input_lengths = torch.full((skeletal.size(0),), skeletal.size(1), dtype=torch.long).to(device)

            # Get predictions (decoded)
            decoded = model(skeletal, crops, optical_flow, input_lengths=input_lengths)

            # Convert decoded predictions and targets into readable strings
            preds = [' '.join([inv_vocab[idx] for idx in seq if idx in inv_vocab]) for seq in decoded]
            targets_str = [' '.join([inv_vocab[idx] for idx in seq if idx != -1]) for seq in targets]

            all_preds.extend(preds)
            all_targets.extend(targets_str)

    # # Compute metrics
    # accuracy = compute_accuracy(all_preds, all_targets)
    # wer_score = compute_wer(all_preds, all_targets)
    # bleu_scores = [compute_bleu(pred.split(), target.split()) for pred, target in zip(all_preds, all_targets)]
    # avg_bleu = np.mean(bleu_scores)

    # # Output results
    # print(f"Test Accuracy: {accuracy:.4f}")
    # print(f"Word Error Rate (WER): {wer_score:.4f}")
    # print(f"Average BLEU Score: {avg_bleu:.4f}")

    print(all_preds)
    print("\n")
    print(all_targets)
