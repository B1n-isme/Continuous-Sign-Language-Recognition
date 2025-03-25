import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import os
import sys
import logging
from tqdm import tqdm
from jiwer import wer

# Append project root for local module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.ema import EMA, get_decay
from src.models.model import CSLRModel  # Your CSLR model definition
from src.training.csl_dataset import CSLDataset, collate_fn  # Dataset and collation utilities
from src.utils.label_utils import build_vocab, load_labels  # Vocabulary utilities
from src.utils.config_loader import load_config
from src.utils.label_utils import decode_targets

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Device selection: use CUDA if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Enable DataLoader pin_memory only for CUDA
pin_memory = True if device.type == "cuda" else False

def train_one_epoch(model, train_loader, optimizer, ema, device, grad_accum_steps=4):
    model.train()
    scaler = GradScaler()
    epoch_loss = 0.0

    # Lists to retain loss graphs (for further plotting and analysis)
    loss_history = []
    acc_history = []  # Optionally, add accuracy or other metrics if available

    optimizer.zero_grad()
    for i, batch in enumerate(tqdm(train_loader, desc="Training"), 1):
        # Move data to device
        skeletal = batch["skeletal"].to(device)
        crops = batch["crops"].to(device)
        optical_flow = batch["optical_flow"].to(device)
        targets = batch["targets"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        # Check dataset for NaN/Inf
        for name, tensor in batch.items():
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                logging.error(f"NaN or Inf found in {name}")
                raise ValueError(f"Invalid values in {name}")

        # Mixed precision forward pass
        with autocast():
            total_loss, enstim_loss, temporal_loss, transformer_loss = model(
                skeletal, crops, optical_flow, targets, input_lengths, target_lengths
            )
        
        # Append loss value for logging (detach to avoid memory leak)
        loss_history.append(total_loss.detach().cpu().item())
        epoch_loss += total_loss.item()

        # Backpropagation with gradient scaling for AMP
        scaler.scale(total_loss).backward()

        # Update EMA after every batch's gradient accumulation
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        ema.update()  # Should be outside the if-block

        # Then control step execution via gradient accumulation
        if i % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    avg_loss = epoch_loss / len(train_loader)
    return avg_loss, loss_history, acc_history  # acc_history can be populated if using accuracy

def validate_one_epoch(model, val_loader, ema, device, lm_path, beam_size=10, lm_weight=0.5):
    model.eval()
    val_loss = 0.0
    all_pred_sentences = []
    all_ref_sentences = []
    loss_history = []  # For validation loss graph
    ema.apply_shadow()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            skeletal = batch["skeletal"].to(device)
            crops = batch["crops"].to(device)
            optical_flow = batch["optical_flow"].to(device)
            targets = batch["targets"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            
            with autocast():
                total_loss, _, _, _ = model(
                    skeletal, crops, optical_flow, targets, input_lengths, target_lengths
                )
            val_loss += total_loss.item()
            loss_history.append(total_loss.detach().cpu().item())
            
            # Decode predictions for WER computation
            # normal decode
            # predictions = model.decode(skeletal, crops, optical_flow, input_lengths)
            # decode with linguistic context
            predictions = model.decode_with_lm(
                skeletal, crops, optical_flow, input_lengths,
                lm_path=lm_path, beam_size=beam_size, lm_weight=lm_weight
            )
            for pred in predictions:
                all_pred_sentences.append(" ".join(pred))

            # Build reference sentences using model.idx_to_gloss
            ref_sentences = decode_targets(targets, target_lengths, model.idx_to_gloss)
            all_ref_sentences.extend(ref_sentences)

    ema.restore()
    avg_val_loss = val_loss / len(val_loader)
    # Compute corpus-level WER
    val_wer = wer(" ".join(all_ref_sentences), " ".join(all_pred_sentences))
    return avg_val_loss, loss_history, val_wer

if __name__ == "__main__":
    # Load configurations
    data_config = load_config("configs/data_config.yaml")
    model_config = load_config("configs/model_config.yaml")
    
    # Dataset paths and hyperparameters
    train_dataset_path = os.path.join(data_config["paths"]["dataset"], "train_dataset.pt")
    val_dataset_path = os.path.join(data_config["paths"]["dataset"], "val_dataset.pt")
    labels_csv = data_config["paths"]["labels"]
    batch_size = model_config["train_params"]["batch_size"]
    learning_rate = model_config["optimize_params"]["learning_rate"]
    weight_decay = model_config["optimize_params"]["weight_decay"]
    num_epochs = model_config["train_params"]["num_epochs"]
    patience = model_config["train_params"]["patience"]
    num_workers = model_config["sys_params"]["num_workers"]
    save_dir = model_config["checkpoint"]["path"]
    os.makedirs(save_dir, exist_ok=True)
    
    # Load datasets
    train_dataset = torch.load(train_dataset_path, weights_only=False)
    val_dataset = torch.load(val_dataset_path, weights_only=False)
    
    # Build vocabulary
    label_dict = load_labels(labels_csv)
    vocab = build_vocab(label_dict)
    vocab_size = len(vocab)
    logging.info(f"Vocabulary size: {vocab_size}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory
    )
    logging.info("Datasets and DataLoaders successfully loaded!")

    # Model parameter
    spatial_params = model_config["spatial_params"]
    model_config["temporal_params"]["vocab_size"] = vocab_size
    temporal_params = model_config["temporal_params"]
    model_config["transformer_params"]["vocab_size"] = vocab_size
    transformer_params = model_config["transformer_params"]
    model_config["enstim_params"]["vocab_size"] = vocab_size
    enstim_params = model_config["enstim_params"]
    
    # Initialize model
    model = CSLRModel(
        spatial_params=spatial_params,
        temporal_params=temporal_params,
        transformer_params=transformer_params,
        enstim_params=enstim_params,
        device=device
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=patience, factor=0.5)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-2,                # Peak learning rate during training
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
        pct_start=0.3,              # Fraction of cycle increasing learning rate
        anneal_strategy='cos',      # Cosine annealing for the decreasing phase
        div_factor=25.0             # Initial lr = max_lr/div_factor
    )
    ema = EMA(model, decay=0.999)
    
    best_val_wer = float("inf")
    patience_counter = 0

    # For plotting training and validation loss graphs over epochs
    history = {"train_loss": [], "val_loss": [], "val_wer": []}
    
    # LM configuration for decoding
    lm_path = "models/checkpoints/kenlm.binary"
    grad_accum_steps = model_config["train_params"]["grad_accum_steps"]

    for epoch in range(num_epochs):
        logging.info(f"Starting epoch {epoch+1}/{num_epochs}")
        
        current_decay = get_decay(epoch, num_epochs)
        ema.decay = current_decay

        # Training phase
        avg_train_loss, train_loss_history, _ = train_one_epoch(
            model, train_loader, optimizer, ema, device, grad_accum_steps=grad_accum_steps
        )
        history["train_loss"].append(avg_train_loss)
        
        # Validation phase
        avg_val_loss, val_loss_history, val_wer = validate_one_epoch(
            model, val_loader, ema, device, lm_path, beam_size=10, lm_weight=0.5
        )
        history["val_loss"].append(avg_val_loss)
        history["val_wer"].append(val_wer)
        
        logging.info(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, Val WER: {val_wer:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}"
        )
        
        # Checkpointing and early stopping based on validation WER
        if val_wer < best_val_wer:
            best_val_wer = val_wer
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_wer': best_val_wer,
                'ema_shadow': ema.shadow,
                'ema_decay': ema.decay
            }
            torch.save(checkpoint, os.path.join(save_dir, f"best_model_epoch{epoch+1}_wer{val_wer:.4f}.pt"))
            logging.info(f"Saved best model with Val WER: {best_val_wer:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            logging.info(f"Patience counter: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            logging.info("Early stopping triggered")
            break
        
        scheduler.step(avg_val_loss)
    
    logging.info("Training completed!")
    # Optionally, save loss history to file for further analysis/plotting
    torch.save(history, os.path.join(save_dir, "loss_history.pt"))
