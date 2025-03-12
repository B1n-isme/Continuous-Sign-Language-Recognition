import torch
from torch.utils.data import DataLoader
from models.cslr_model import CSLRModel
from data.csl_dataset import CSLDataset, collate_fn
from utils.labels_utils import build_vocab, load_labels
from jiwer import wer  # For Word Error Rate
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from tqdm import tqdm

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
test_dataset_path = "data/datasets/test_dataset.pt"
labels_csv = "data/processed_labels.csv"
model_path = "checkpoints/best_model.pth"

# Load test dataset and vocabulary
test_dataset = torch.load(test_dataset_path)
label_dict = load_labels(labels_csv)
vocab = build_vocab(label_dict)
vocab_size = len(vocab)

# Initialize and load model
model = CSLRModel(vocab_size=vocab_size, blank=0).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Set up DataLoader
batch_size = 4
test_loader = DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    collate_fn=collate_fn, 
    num_workers=2
)

# Metric functions
def compute_accuracy(preds, targets):
    correct = sum([1 for p, t in zip(preds, targets) if p == t])
    return correct / len(targets)

def compute_wer(preds, targets):
    return wer(targets, preds)

def compute_bleu(pred, target):
    return sentence_bleu([target], pred)

# Evaluation loop
all_preds = []
all_targets = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        skeletal = batch['skeletal'].to(device)
        crops = batch['crops'].to(device)
        optical_flow = batch['optical_flow'].to(device)
        targets = batch['labels'].cpu().numpy()
        input_lengths = torch.full((batch_size,), skeletal.size(1), dtype=torch.long).to(device)
        
        # Get predictions
        decoded = model(skeletal, crops, optical_flow, input_lengths=input_lengths)
        
        # Convert to readable strings
        inv_vocab = {v: k for k, v in vocab.items()}
        preds = [' '.join([inv_vocab[idx] for idx in seq if idx in inv_vocab]) for seq in decoded]
        targets = [' '.join([inv_vocab[idx] for idx in seq if idx != -1]) for seq in targets]
        
        all_preds.extend(preds)
        all_targets.extend(targets)

# Compute metrics
accuracy = compute_accuracy(all_preds, all_targets)
wer_score = compute_wer(all_preds, all_targets)
bleu_scores = [compute_bleu(pred.split(), target.split()) for pred, target in zip(all_preds, all_targets)]
avg_bleu = np.mean(bleu_scores)

# Output results
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Word Error Rate (WER): {wer_score:.4f}")
print(f"Average BLEU Score: {avg_bleu:.4f}")