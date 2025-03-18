import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
from tqdm import tqdm
import logging
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
    # Paths and hyperparameters
    train_dataset_path = "data/datasets/train_dataset.pt"
    val_dataset_path = "data/datasets/val_dataset.pt"
    labels_csv = "data/labels.csv"
    save_dir = "checkpoints/"
    batch_size = 4
    num_epochs = 50
    learning_rate = 1e-3
    weight_decay = 1e-5
    patience = 5  # For early stopping

    # Create checkpoint directory
    os.makedirs(save_dir, exist_ok=True)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load datasets
    try:
        train_dataset = torch.load(train_dataset_path, weights_only=False)
        val_dataset = torch.load(val_dataset_path, weights_only=False)
    except Exception as e:
        logging.error(f"Failed to load datasets: {e}")
        raise

    # Build vocabulary from labels
    label_dict = load_labels(labels_csv)  # Replace with your implementation
    vocab = build_vocab(label_dict)       # Replace with your implementation
    vocab_size = len(vocab)
    logging.info(f"Vocabulary size: {vocab_size}")

    # Set up DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,  # Ensure this handles padding and creates targets tensor
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
    logging.info("Successfully loaded datasets and created DataLoaders!")

    # Define model parameters (as per model.py example)
    spatial_params = {"D_spatial": 128}
    temporal_params = {
        "in_channels": 128,
        "out_channels": 256,
        "kernel_sizes": [3, 5, 7],
        "dilations": [1, 2, 4],
        "vocab_size": vocab_size
    }
    transformer_params = {
        "input_dim": 2 * 256,  # 2 hands * 256 features
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

    # Initialize the CSLRModel
    model = CSLRModel(
        spatial_params=spatial_params,
        temporal_params=temporal_params,
        transformer_params=transformer_params,
        enstim_params=enstim_params,
        device=device
    ).to(device)

    # Set up optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    # Training loop with early stopping
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        ### Training Phase
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
            # Move batch data to device
            skeletal = batch["skeletal"].to(device)
            crops = batch["crops"].to(device)
            optical_flow = batch["optical_flow"].to(device)
            targets = batch["targets"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            target_lengths = batch["target_lengths"].to(device)

            # Forward pass
            optimizer.zero_grad()
            total_loss, enstim_loss, temporal_loss, transformer_loss = model(
                skeletal, crops, optical_flow, targets, input_lengths, target_lengths
            )
            
            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()

        avg_train_loss = train_loss / len(train_loader)
        
        ### Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation"):
                skeletal = batch["skeletal"].to(device)
                crops = batch["crops"].to(device)
                optical_flow = batch["optical_flow"].to(device)
                targets = batch["targets"].to(device)
                input_lengths = batch["input_lengths"].to(device)
                target_lengths = batch["target_lengths"].to(device)

                total_loss, _, _, _ = model(
                    skeletal, crops, optical_flow, targets, input_lengths, target_lengths
                )
                val_loss += total_loss.item()

        avg_val_loss = val_loss / len(val_loader)
        
        # Logging
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping and checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
            logging.info(f"Saved best model with Val Loss: {best_val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            logging.info(f"Patience counter: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            logging.info("Early stopping triggered")
            break
        
        # Adjust learning rate based on validation loss
        scheduler.step(avg_val_loss)

    logging.info("Training completed!")