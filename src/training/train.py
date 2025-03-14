import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import logging
import sys
from jiwer import wer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.model import CSLRModel  # Your CSLR model definition
from src.training.csl_dataset import (
    CSLDataset,
    collate_fn,
)  # Dataset and collation utilities
from src.utils.label_utils import build_vocab, load_labels  # Vocabulary utilities

# Set up logging and device
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

if __name__ == "__main__":
    # Paths to datasets and labels
    train_dataset_path = "data/datasets/train_dataset.pt"
    val_dataset_path = "data/datasets/val_dataset.pt"
    labels_csv = "data/labels.csv"
    save_dir = "checkpoints/"
    os.makedirs(save_dir, exist_ok=True)

    # Load datasets
    try:
        train_dataset = torch.load(train_dataset_path, weights_only=False)
        val_dataset = torch.load(val_dataset_path, weights_only=False)
    except Exception as e:
        logging.error(f"Failed to load datasets: {e}")
        raise

    # Build vocabulary from labels
    label_dict = load_labels(labels_csv)
    vocab = build_vocab(label_dict)
    vocab_size = len(vocab)
    logging.info(f"Vocabulary size: {vocab_size}")

    # Set up DataLoaders
    batch_size = 4
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        drop_last=True
    )
    logging.info("Successfully loaded datasets!")

    # Initialize the CSLRModel
    model = CSLRModel(vocab_size=vocab_size, blank=0, device=device).to(device)

    # Set up optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    # Training loop with early stopping
    num_epochs = 50
    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            skeletal = batch["skeletal"].to(device)
            crops = batch["crops"].to(device)
            optical_flow = batch["optical_flow"].to(device)
            targets = batch["targets"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            target_lengths = batch["target_lengths"].to(device)  # Assumes -1 padding

            # Forward pass
            loss = model(skeletal, crops, optical_flow, targets, input_lengths, target_lengths)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        logging.info(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        if wer:
            val_wer = 0.0
            total_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                skeletal = batch["skeletal"].to(device)
                crops = batch["crops"].to(device)
                optical_flow = batch["optical_flow"].to(device)
                targets = batch["targets"].to(device)  # Now (sum(L_i),)
                input_lengths = batch["input_lengths"].to(device)
                target_lengths = batch["target_lengths"].to(device)  # Use directly from collate_fn, shape (B,)

                # Compute loss
                loss = model(skeletal, crops, optical_flow, targets, input_lengths, target_lengths)
                val_loss += loss.item()

                # Optional WER calculation
                if wer:
                    decoded = model(skeletal, crops, optical_flow, input_lengths=input_lengths)
                    # Convert concatenated targets back to list of sequences
                    start_idx = 0
                    ground_truth = []
                    for tl in target_lengths:
                        seq = targets[start_idx:start_idx + tl.item()].tolist()
                        ground_truth.append([g for g in seq if g != 0])  # Remove blank if present
                        start_idx += tl.item()
                    decoded_clean = [[g for g in seq if g != 0] for seq in decoded]  # Remove blank token
                    val_wer += wer(ground_truth, decoded_clean) * batch_size
                    total_samples += batch_size

        val_loss /= len(val_loader)
        val_metrics = f"Epoch {epoch + 1}, Val Loss: {val_loss:.4f}"
        if wer:
            val_wer /= total_samples
            val_metrics += f", Val WER: {val_wer:.4f}"
        logging.info(val_metrics + f", LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Adjust learning rate
        scheduler.step(val_loss)

        # Save best model and implement early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch + 1,
                "val_loss": val_loss
            }
            torch.save(checkpoint, os.path.join(save_dir, "best_model.pth"))
            logging.info(f"Model saved at epoch {epoch + 1} with Val Loss: {val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

    logging.info("Training completed!")