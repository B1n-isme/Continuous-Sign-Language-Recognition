import os
import sys
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Load the loss history from the .pt file
loss_history = torch.load("all_working/checkpoints/loss_history.pt")

# Check if the loaded object is a dictionary.
# If so, try to extract the loss list from a key named "loss".
if isinstance(loss_history, dict):
    if "loss" in loss_history:
        losses = loss_history["loss"]
    else:
        # If "loss" key is not present, use the first key's value as a fallback.
        losses = list(loss_history.values())[0]
else:
    losses = loss_history

# Create a plot for the training loss
plt.figure(figsize=(8, 5))
plt.plot(losses, marker='o', linestyle='-', color='b')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss History")
plt.grid(True)
plt.show()
