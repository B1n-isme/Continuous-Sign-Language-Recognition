import numpy as np
import pandas as pd
import os
import glob
from scipy.interpolate import interp1d

from skeletal_aug import (
    interpolate_skeletal,
    smooth_skeletal_ema,
    rotate_skeletal,
    time_warp_skeletal,
    add_gaussian_noise_skeletal,
    frame_drop_skeletal,
    scale_skeletal,
)
from crop_aug import normalize_crops, scale_crops, rotate_crops, occlude_fingers_crops
from flow_aug import normalize_optical_flow


def align_and_preprocess_modalities(
    skeletal_variant, crops, optical_flow, variant_name
):
    """
    Align and preprocess different modalities to have the same temporal dimension.

    Args:
        skeletal_variant (np.ndarray): Skeletal data after augmentation.
        crops (np.ndarray): Original cropped images.
        optical_flow (np.ndarray): Original optical flow data.
        variant_name (str): Name of the augmentation variant.

    Returns:
        tuple: Aligned and preprocessed crops and optical flow data.
    """
    T_new = skeletal_variant.shape[0]
    T_orig = crops.shape[0]
    old_frames = np.arange(T_orig)
    new_frames = np.linspace(0, T_orig - 1, T_new)

    crops_normalized = normalize_crops(crops)
    if "rotated_15" in variant_name:
        crops_variant = rotate_crops(crops_normalized, 15)
        crops_variant = scale_crops(crops_variant, 1.2)
    elif "rotated_-15" in variant_name:
        crops_variant = rotate_crops(crops_normalized, -15)
        crops_variant = scale_crops(crops_variant, 0.8)
    elif "noisy" in variant_name:
        crops_variant = occlude_fingers_crops(crops_normalized)
    else:
        crops_variant = crops_normalized

    crops_new = np.zeros((T_new, *crops_variant.shape[1:]), dtype=np.float32)
    for h in range(2):
        for c in range(3):
            interp = interp1d(old_frames, crops_variant[:, h, :, :, c], axis=0)
            crops_new[:, h, :, :, c] = interp(new_frames)

    flow_normalized = normalize_optical_flow(optical_flow)
    T_flow_orig = optical_flow.shape[0]
    new_flow_frames = np.linspace(0, T_flow_orig - 1, T_new - 1)
    flow_new = np.zeros((T_new - 1, *flow_normalized.shape[1:]), dtype=np.float32)
    for h in range(2):
        for c in range(2):
            interp = interp1d(
                np.arange(T_flow_orig), flow_normalized[:, h, :, :, c], axis=0
            )
            flow_new[:, h, :, :, c] = interp(new_flow_frames)

    return crops_new, flow_new


def load_data(file_path):
    """
    Load data from NPZ file.

    Args:
        file_path (str): Path to the data file.

    Returns:
        tuple: Skeletal data, crops, optical flow, and labels.
    """
    data = np.load(file_path, allow_pickle=True)
    skeletal_data = data["skeletal_data"]
    crops = data["crops"]
    optical_flow = data["optical_flow"]
    labels = data["labels"]

    return skeletal_data, crops, optical_flow, labels


def generate_variants(skeletal_data):
    """
    Generate different variants of skeletal data.

    Args:
        skeletal_data (np.ndarray): Original skeletal data.

    Returns:
        list: List of tuples containing (variant_name, augmented_data).
    """
    cleaned_skeletal = smooth_skeletal_ema(interpolate_skeletal(skeletal_data))

    variants = [
        ("original", cleaned_skeletal),
        (
            "rotated_15",
            rotate_skeletal(
                scale_skeletal(cleaned_skeletal, np.random.uniform(0.9, 1.1))
            ),
        ),
        (
            "rotated_-15",
            rotate_skeletal(
                scale_skeletal(cleaned_skeletal, np.random.uniform(0.9, 1.1))
            ),
        ),
        ("warped_slow", time_warp_skeletal(cleaned_skeletal, 1.2)),
        ("warped_fast", time_warp_skeletal(cleaned_skeletal, 0.8)),
        ("noisy", add_gaussian_noise_skeletal(cleaned_skeletal, 0.01)),
        ("frame_dropped", frame_drop_skeletal(cleaned_skeletal, 0.1)),
    ]

    return variants


def process_file(file_path, output_dir):
    """
    Process a single data file and generate augmented variants.

    Args:
        file_path (str): Path to the data file.
    """
    print(f"Processing {os.path.basename(file_path)}...")

    # Load data
    skeletal_data, crops, optical_flow, labels = load_data(file_path)

    # Generate variants
    variants = generate_variants(skeletal_data)

    # Base name for output files
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    # Process and save each variant
    for name, skeletal_variant in variants:
        crops_aligned, flow_aligned = align_and_preprocess_modalities(
            skeletal_variant, crops, optical_flow, name
        )

        output_path = os.path.join(output_dir, f"{base_name}_{name}.npz")

        np.savez(
            output_path,
            skeletal_data=skeletal_variant,
            crops=crops_aligned,
            optical_flow=flow_aligned,
            labels=labels,
        )

        print(
            f"  Saved {name} variant - Skeletal: {skeletal_variant.shape}, "
            f"Crops: {crops_aligned.shape}, Flow: {flow_aligned.shape}"
        )


def main():
    """Main function to process all raw data files."""
    output_dir = "data/processed"
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all .npz files in the raw data directory
    raw_files = glob.glob("data/raw/*.npz")

    if not raw_files:
        print("No .npz files found in data/raw/ directory.")
        return

    print(f"Found {len(raw_files)} files to process.")

    # Process each file
    for file_path in raw_files:
        process_file(file_path, output_dir)

    print("All files processed successfully.")


if __name__ == "__main__":
    main()
