import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import sys
import cv2
from pathlib import Path
import h5py

def load_h5_file(filepath):
    """
    Load and return an HDF5 file object.

    Args:
        filepath: Path to the .h5 file

    Returns:
        h5py.File: Open HDF5 file object in read mode
    """
    try:
        f = h5py.File(filepath, "r")
        return f
    except Exception as e:
        print(f"Error loading file {filepath}: {e}")
        return None

def display_h5_info(f):
    """
    Display information about the contents of an HDF5 file.

    Args:
        f: HDF5 file object
    """
    if f is None:
        return

    print("\n===== H5 File Contents =====")
    keys = list(f.keys())
    print(f"Available keys: {keys}")

    for key in keys:
        dataset = f[key]
        print(f"\nKey: {key}")
        print(f"  Type: {type(dataset)}")
        print(f"  Shape: {dataset.shape}")
        print(f"  Data type: {dataset.dtype}")

        if key == "skeletal_data":
            print(f"  Frames: {dataset.shape[0]}")
            print(f"  Max hands: {dataset.shape[1]}")
            print(f"  Landmarks: {dataset.shape[2]}")
            print(f"  Coordinates: {dataset.shape[3]}")
        elif key == "crops":
            print(f"  Crop dataset shape: {dataset.shape}")
            # If you stored crops as (T, H, height, width, C), you can analyze further if needed.
        elif key == "labels":
            labels = dataset[()]
            print(f"  Labels: {labels}")
        elif key == "crop_size":
            print(f"  Value: {dataset[()]}")
        elif key == "max_hands":
            print(f"  Value: {dataset[()]}")

def visualize_sample(f, show_frames=None, animated=True, pause_time=0.2):
    """
    Visualize sample data from the HDF5 file.

    Args:
        f: HDF5 file object
        show_frames: Number of sample frames to visualize (None for all frames)
        animated: Whether to display as an animation (True) or as static plots (False)
        pause_time: Pause time between frames for animation
    """
    if f is None or "crops" not in f or "skeletal_data" not in f:
        return

    # Load datasets into memory
    crops = f["crops"][()]
    skeletal = f["skeletal_data"][()]

    # Identify valid frames based on crops (non-empty frames)
    valid_frames = [i for i, item in enumerate(crops) if item.size > 0]

    if not valid_frames:
        print("No valid frames with hand crops found.")
        return

    if show_frames is not None and not animated:
        sample_indices = valid_frames[:min(show_frames, len(valid_frames))]
    else:
        sample_indices = valid_frames

    fig = plt.figure(figsize=(15, 5))

    for frame_idx in sample_indices:
        if animated and frame_idx != sample_indices[0]:
            plt.clf()

        frame_crops = crops[frame_idx]
        # Assume frame_crops has shape (num_hands, height, width, channels)
        num_crops = frame_crops.shape[0] if frame_crops.ndim > 0 else 0

        for i in range(num_crops):
            plt.subplot(1, num_crops + 1, i + 1)
            crop = frame_crops[i]
            plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            plt.title(f"Hand {i+1}")
            plt.axis("off")

        plt.subplot(1, num_crops + 1, num_crops + 1)
        frame_skeletal = skeletal[frame_idx]
        # Visualize up to 2 hands
        for hand_idx in range(min(frame_skeletal.shape[0], 2)):
            hand_data = frame_skeletal[hand_idx]
            x_coords = hand_data[:, 0]
            y_coords = hand_data[:, 1]
            # Skip if all zeros (indicating no hand)
            if np.all(x_coords == 0) and np.all(y_coords == 0):
                continue
            color = "r" if hand_idx == 0 else "g"
            plt.scatter(x_coords, y_coords, c=color, label=f"Hand {hand_idx+1}")
            # Connect keypoints (simplified example)
            for finger in range(5):
                base = 1 + finger * 4
                points = [0, base, base + 1, base + 2, base + 3]
                xs = [x_coords[i] for i in points if i < len(x_coords)]
                ys = [y_coords[i] for i in points if i < len(y_coords)]
                plt.plot(xs, ys, color)
        plt.title("Skeletal Data")
        plt.gca().invert_yaxis()  # Invert Y-axis for proper image alignment
        plt.axis("equal")
        plt.legend()
        plt.tight_layout()
        plt.suptitle(f"Frame {frame_idx} / {len(crops)}")

        if animated:
            plt.pause(pause_time)
        else:
            plt.show()

    if animated:
        plt.show()

def check_crop_value_range(crops):
    """
    Check the value range of images in the 'crops' array to determine if they are normalized.

    Args:
        crops (np.ndarray): Array containing image data.
    """
    if not isinstance(crops, np.ndarray):
        print("Error: 'crops' is not a NumPy array.")
        return

    pixel_values = []
    for frame in crops:
        if frame.size > 0:
            for crop in frame:
                pixel_values.append(crop.flatten())
    if not pixel_values:
        print("No valid images found in 'crops'.")
        return
    pixel_values = np.concatenate(pixel_values)
    print(f"Min pixel value: {np.min(pixel_values)}")
    print(f"Max pixel value: {np.max(pixel_values)}")
    if np.min(pixel_values) >= 0 and np.max(pixel_values) <= 1:
        print("Images are normalized (range: [0,1]).")
    elif np.min(pixel_values) >= -1 and np.max(pixel_values) <= 1:
        print("Images are normalized (range: [-1,1]).")
    elif np.min(pixel_values) >= 0 and np.max(pixel_values) <= 255:
        print("Images have standard 8-bit range (0-255).")
    else:
        print("Images have an unusual value range.")

def main():
    file_path = "data/raw/Hello-I-Have-Good-Lunch_1dujA7.h5"
    f = load_h5_file(file_path)
    display_h5_info(f)
    # To check value range of crops, uncomment the following line:
    # check_crop_value_range(f["crops"][()])
    visualize_sample(f, animated=True, pause_time=0.2)

if __name__ == "__main__":
    main()
