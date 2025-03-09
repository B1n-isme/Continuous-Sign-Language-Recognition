import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

file_path = "data/raw/recording_20250309_210346.npz"
data = np.load(file_path, allow_pickle=True)

# Access the image data
crops = data["crops"]

# If crops is a single image
if len(crops.shape) == 3:  # Height, Width, Channels
    plt.imshow(crops)
    plt.axis('off')
    plt.title("Image from crops")
    plt.show()
# If crops contains multiple images
elif len(crops.shape) == 4:  # Batch, Height, Width, Channels
    # Display the first few images
    n_images = min(4, crops.shape[0])
    fig, axes = plt.subplots(1, n_images, figsize=(15, 5))
    for i in range(n_images):
        if n_images == 1:
            axes.imshow(crops[i])
            axes.axis('off')
            axes.set_title(f"Image {i}")
        else:
            axes[i].imshow(crops[i])
            axes[i].axis('off')
            axes[i].set_title(f"Image {i}")
    plt.tight_layout()
    plt.show()
else:
    print(f"Unexpected crops shape: {crops.shape}")
    print(f"Data type: {crops.dtype}")
