import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
from tqdm import tqdm
import logging
from jiwer import wer

# Append project root for local module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.model import CSLRModel  # Your CSLR model definition
from src.training.csl_dataset import CSLDataset, collate_fn  # Dataset and collation utilities
from src.utils.label_utils import build_vocab, load_labels  # Vocabulary utilities

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Device selection: use CUDA if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Enable DataLoader pin_memory only for CUDA
pin_memory = True if device.type == "cuda" else False

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
    patience = 10  # Early stopping patience

    # Create checkpoint directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Load datasets
    try:
        train_dataset = torch.load(train_dataset_path, weights_only=False)
        val_dataset = torch.load(val_dataset_path, weights_only=False)
    except Exception as e:
        logging.error(f"Failed to load datasets: {e}")
        raise

    # Build vocabulary from labels
    label_dict = load_labels(labels_csv)  # Replace with your implementation
    vocab = build_vocab(label_dict)         # Replace with your implementation
    vocab_size = len(vocab)
    logging.info(f"Vocabulary size: {vocab_size}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,  # Handles padding and target tensor creation
        num_workers=4,
        drop_last=True,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        drop_last=True,
        pin_memory=pin_memory
    )
    logging.info("Successfully loaded datasets and created DataLoaders!")

    # Define model parameters
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

    # Initialize the model and move it to the selected device
    model = CSLRModel(
        spatial_params=spatial_params,
        temporal_params=temporal_params,
        transformer_params=transformer_params,
        enstim_params=enstim_params,
        device=device
    ).to(device)

    # Optionally compile the model for performance improvements (PyTorch 2.0+)
    if device.type == "cuda" and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            logging.info("Model compiled with torch.compile for improved performance.")
        except Exception as e:
            logging.warning(f"torch.compile failed: {e}")

    # Set up the optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            # Move batch data to device
            skeletal = batch["skeletal"].to(device)
            crops = batch["crops"].to(device)
            optical_flow = batch["optical_flow"].to(device)
            targets = batch["targets"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            target_lengths = batch["target_lengths"].to(device)

            optimizer.zero_grad()
            total_loss, enstim_loss, temporal_loss, transformer_loss = model(
                skeletal, crops, optical_flow, targets, input_lengths, target_lengths
            )
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
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
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Checkpointing and early stopping
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

        scheduler.step(avg_val_loss)

    logging.info("Training completed!")
