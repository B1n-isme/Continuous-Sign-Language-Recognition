import numpy as np

# Load the .npz file
data = np.load("data/raw/recording_20250309_210400.npz", allow_pickle=True)

# Extract existing arrays
skeletal_data = data['skeletal_data']
crops = data['crops']
labels = data['labels']

# New array with zeros of shape (105, 2, 112, 112, 3)
new_crops = np.zeros((105, 2, 112, 112, 3), dtype=np.uint8)  # Assuming uint8 for images

# Iterate through the crops and place them in the correct slot
for i, crop in enumerate(crops):
    if crop.shape[0] == 1:
        new_crops[i, 1] = crop[0]  # Assign to the first hand slot
    elif crop.shape[0] == 2:
        new_crops[i] = crop  # Assign directly since both hands exist

# Save the modified fileq
np.savez("data/raw/reshape.npz", skeletal_data=skeletal_data, crops=new_crops, labels=labels)
