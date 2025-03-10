import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import sys
import cv2
from pathlib import Path


def load_npz_file(filepath):
    """
    Load and return contents of an .npz file.

    Args:
        filepath: Path to the .npz file

    Returns:
        dict: Dictionary containing the loaded arrays
    """
    try:
        data = np.load(filepath, allow_pickle=True)
        return data
    except Exception as e:
        print(f"Error loading file {filepath}: {e}")
        return None


def display_npz_info(data):
    """
    Display information about the contents of an .npz file.

    Args:
        data: Loaded .npz data object
    """
    if data is None:
        return

    print("\n===== NPZ File Contents =====")
    print(f"Available keys: {list(data.keys())}")

    # Display information about each key
    for key in data.keys():
        array = data[key]
        print(f"\nKey: {key}")
        print(f"  Type: {type(array)}")
        print(f"  Shape: {array.shape}")
        print(f"  Data type: {array.dtype}")

        # Additional info for specific arrays
        if key == "skeletal_data":
            print(f"  Frames: {array.shape[0]}")
            print(f"  Max hands: {array.shape[1]}")
            print(f"  Landmarks: {array.shape[2]}")
            print(f"  Coordinates: {array.shape[3]}")

        elif key == "crops":
            non_empty = sum(1 for item in array if len(item) > 0)
            print(f"  Frames with hand crops: {non_empty}/{len(array)}")
            
            # Analyze crop shapes in detail
            print(f"  Detailed crop analysis:")
            
            # Count frames with different numbers of crops
            hand_count = {}
            for item in array:
                num_hands = len(item) if isinstance(item, np.ndarray) else 0
                hand_count[num_hands] = hand_count.get(num_hands, 0) + 1
            
            for num, count in sorted(hand_count.items()):
                print(f"    Frames with {num} hands: {count} ({count/len(array)*100:.1f}%)")
            
            # Show actual shapes of a few samples
            if non_empty > 0:
                print("  Sample crop shapes:")
                samples_shown = 0
                for i, item in enumerate(array):
                    if len(item) > 0:
                        print(f"    Frame {i}: {item.shape} - Each crop: ", end="")
                        if len(item.shape) >= 3:  # Has width, height, channels
                            print(f"{item[0].shape if item.shape[0] > 0 else 'No crops'}")
                        else:
                            print(f"{item.shape}")
                        samples_shown += 1
                        # if samples_shown >= 5:  # Show at most 5 samples
                        #     break
                
                # Get all unique crop shapes
                unique_shapes = set()
                for item in array:
                    if len(item) > 0:
                        for crop in item:
                            unique_shapes.add(crop.shape)
                
                print(f"  Unique crop shapes found: {unique_shapes}")

        elif key == "labels":
            if len(array) > 0:
                print(f"  Labels: {array}")

        elif key == "crop_size":
            print(f"  Value: {array}")

        elif key == "max_hands":
            print(f"  Value: {array}")


def visualize_sample(data, show_frames=2):
    """
    Visualize sample data from the .npz file.

    Args:
        data: Loaded .npz data object
        show_frames: Number of sample frames to visualize
    """
    if data is None or "crops" not in data or "skeletal_data" not in data:
        return

    crops = data["crops"]
    skeletal = data["skeletal_data"]

    # Find frames with hand crops
    valid_frames = [i for i, item in enumerate(crops) if len(item) > 0]

    if not valid_frames:
        print("No valid frames with hand crops found.")
        return

    # Select a subset of frames to display
    sample_indices = valid_frames[: min(show_frames, len(valid_frames))]

    for frame_idx in sample_indices:
        fig = plt.figure(figsize=(15, 5))

        # Plot crops
        frame_crops = crops[frame_idx]
        num_crops = len(frame_crops)

        for i, crop in enumerate(frame_crops):
            plt.subplot(1, num_crops + 1, i + 1)
            plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            plt.title(f"Hand {i + 1}")
            plt.axis("off")

        # Plot skeletal data
        plt.subplot(1, num_crops + 1, num_crops + 1)
        frame_skeletal = skeletal[frame_idx]

        # Create a scatter plot of the first hand's landmarks
        for hand_idx in range(min(frame_skeletal.shape[0], 2)):  # Show max 2 hands
            hand_data = frame_skeletal[hand_idx]
            x_coords = hand_data[:, 0]
            y_coords = hand_data[:, 1]

            # Skip if all zeros (no hand)
            if np.all(x_coords == 0) and np.all(y_coords == 0):
                continue

            color = "r" if hand_idx == 0 else "g"
            plt.scatter(x_coords, y_coords, c=color, label=f"Hand {hand_idx + 1}")

            # Connect key points with lines (simplified)
            for finger in range(5):
                base = 1 + finger * 4  # Base of each finger
                points = [0, base, base + 1, base + 2, base + 3]  # Connect to wrist
                xs = [x_coords[i] for i in points]
                ys = [y_coords[i] for i in points]
                plt.plot(xs, ys, color)

        plt.title("Skeletal Data")
        plt.gca().invert_yaxis()  # Invert Y-axis to match image coordinates
        plt.axis("equal")
        plt.legend()

        plt.tight_layout()
        plt.suptitle(f"Frame {frame_idx}")
        plt.show()


def main():
    """Parse arguments and check npz file."""
    parser = argparse.ArgumentParser(
        description="Check the shape and content of .npz files"
    )
    parser.add_argument("filepath", help="Path to the .npz file")
    parser.add_argument(
        "--visualize", "-v", action="store_true", help="Visualize sample data"
    )
    parser.add_argument(
        "--frames",
        "-f",
        type=int,
        default=2,
        help="Number of frames to visualize (if --visualize is set)",
    )

    args = parser.parse_args()

    # Ensure the file exists
    if not os.path.exists(args.filepath):
        print(f"Error: File {args.filepath} does not exist")
        sys.exit(1)

    # Load and display file info
    data = load_npz_file(args.filepath)
    if data is not None:
        display_npz_info(data)

        if args.visualize:
            visualize_sample(data, args.frames)


if __name__ == "__main__":
    main()
