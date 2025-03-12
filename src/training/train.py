import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import logging
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.model import CSLRModel  # Your CSLR model definition
from training.csl_dataset import (
    CSLDataset,
    collate_fn,
)  # Dataset and collation utilities
from utils.label_utils import build_vocab, load_labels  # Vocabulary utilities

# Set up logging and device
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    # Paths to datasets and labels
    train_dataset_path = "data/datasets/train_dataset.pt"
    val_dataset_path = "data/datasets/val_dataset.pt"
    labels_csv = "data/labels.csv"

    # Load datasets
    train_dataset = torch.load(train_dataset_path, weights_only=False)
    val_dataset = torch.load(val_dataset_path, weights_only=False)

    # Build vocabulary from labels
    label_dict = load_labels(labels_csv)
    vocab = build_vocab(label_dict)
    vocab_size = len(vocab)

    # Set up DataLoaders
    batch_size = 4
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    print("Successfully loaded datasets!")

    # Initialize the CSLRModel
    model = CSLRModel(
        vocab_size=vocab_size,
        blank=0,  # Index for blank token in CTC
        lambda_entropy=0.1,
        D_skeletal=64,
        D_cnn=512,
        D_flow=512,
        D_temp=256,
        model_dim=512,
        num_heads=8,
        num_layers=4,
        hidden_dim_rnn=256,
    ).to(device)

    # Set up optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    # Training loop
    num_epochs = 50
    best_val_loss = float("inf")
    save_dir = "checkpoints/"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            skeletal = batch["skeletal"].to(device)
            crops = batch["crops"].to(device)
            optical_flow = batch["optical_flow"].to(device)
            targets = batch["labels"].to(device)

            # Dynamic batch size handling
            current_batch_size = skeletal.size(0)
            input_lengths = torch.full(
                (current_batch_size,), skeletal.size(1), dtype=torch.long
            ).to(device)
            target_lengths = torch.sum(targets != -1, dim=1).to(device)  # -1 as padding

            # Forward pass
            loss = model(
                skeletal, crops, optical_flow, targets, input_lengths, target_lengths
            )

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        logging.info(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                skeletal = batch["skeletal"].to(device)
                crops = batch["crops"].to(device)
                optical_flow = batch["optical_flow"].to(device)
                targets = batch["labels"].to(device)

                # Dynamic batch size handling
                current_batch_size = skeletal.size(0)
                input_lengths = torch.full(
                    (current_batch_size,), skeletal.size(1), dtype=torch.long
                ).to(device)
                target_lengths = torch.sum(targets != -1, dim=1).to(device)

                loss = model(
                    skeletal,
                    crops,
                    optical_flow,
                    targets,
                    input_lengths,
                    target_lengths,
                )
                val_loss += loss.item()

        val_loss /= len(val_loader)
        logging.info(f"Epoch {epoch + 1}, Val Loss: {val_loss:.4f}")

        # Adjust learning rate
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            logging.info(f"Model saved at epoch {epoch + 1}")
