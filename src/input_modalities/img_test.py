import numpy as np
import matplotlib.pyplot as plt

# Load the modified .npz file
data = np.load("data/raw/output_with_flow.npz", allow_pickle=True)
crops = data['crops']  # Shape: (105, 2, 112, 112, 3)

# Number of frames to display
num_frames = len(crops)  # Ensure we don't exceed available frames

# Create a figure with 2 subplots (left & right hand)
fig, axes = plt.subplots(1, 2, figsize=(6, 3))

for i in range(num_frames):
    left_hand, right_hand = crops[i]  # Extract left and right hand crops
    
    # Update plots
    axes[0].imshow(left_hand.astype(np.uint8))
    axes[0].set_title(f"Frame {i+1} - Left Hand")
    axes[0].axis("off")

    axes[1].imshow(right_hand.astype(np.uint8))
    axes[1].set_title(f"Frame {i+1} - Right Hand")
    axes[1].axis("off")

    plt.pause(0.1)  # Pause to create an animation effect
    # plt.clf()  # Clear figure before next frame

plt.show()
