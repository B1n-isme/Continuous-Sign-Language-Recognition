import numpy as np
from scipy.interpolate import interp1d


from skeletal_aug import (interpolate_skeletal, smooth_skeletal_ema, rotate_skeletal,
                           time_warp_skeletal, add_gaussian_noise_skeletal, frame_drop_skeletal,
                           scale_skeletal)
from crop_aug import normalize_crops, scale_crops, rotate_crops, occlude_fingers_crops
from flow_aug import normalize_optical_flow


# Align modalities (updated to include preprocessing)
def align_and_preprocess_modalities(skeletal_variant, crops, optical_flow, variant_name):
    T_new = skeletal_variant.shape[0]
    T_orig = crops.shape[0]
    old_frames = np.arange(T_orig)
    new_frames = np.linspace(0, T_orig-1, T_new)
    
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
    new_flow_frames = np.linspace(0, T_flow_orig-1, T_new-1)
    flow_new = np.zeros((T_new-1, *flow_normalized.shape[1:]), dtype=np.float32)
    for h in range(2):
        for c in range(2):
            interp = interp1d(np.arange(T_flow_orig), flow_normalized[:, h, :, :, c], axis=0)
            flow_new[:, h, :, :, c] = interp(new_flow_frames)
    
    return crops_new, flow_new

# Load and process
data = np.load("data/raw/output_with_flow.npz", allow_pickle=True)
skeletal_data = data["skeletal_data"]  # (104, 2, 21, 3) or (105, 2, 21, 3)
crops = data["crops"]
optical_flow = data["optical_flow"]
labels = data["labels"]

# if skeletal_data.shape[0] == 105:
#     skeletal_data = skeletal_data[:-1]
#     crops = crops[:-1]
#     optical_flow = optical_flow

# Generate and save 7 variants
cleaned_skeletal = smooth_skeletal_ema(interpolate_skeletal(skeletal_data))
variants = [
    ("original", cleaned_skeletal),
    ("rotated_15", rotate_skeletal(scale_skeletal(cleaned_skeletal, np.random.uniform(0.9, 1.1)))),
    ("rotated_-15", rotate_skeletal(scale_skeletal(cleaned_skeletal, np.random.uniform(0.9, 1.1)))),
    ("warped_slow", time_warp_skeletal(cleaned_skeletal, 1.2)),
    ("warped_fast", time_warp_skeletal(cleaned_skeletal, 0.8)),
    ("noisy", add_gaussian_noise_skeletal(cleaned_skeletal, 0.01)),
    ("frame_dropped", frame_drop_skeletal(cleaned_skeletal, 0.1))
]

for name, skeletal_variant in variants:
    crops_aligned, flow_aligned = align_and_preprocess_modalities(skeletal_variant, crops, optical_flow, name)
    np.savez(f"data/preprocessed/video1_{name}.npz",
             skeletal_data=skeletal_variant,
             crops=crops_aligned,
             optical_flow=flow_aligned,
             labels=labels)
    print(f"Saved {name} - Skeletal: {skeletal_variant.shape}, Crops: {crops_aligned.shape}, Flow: {flow_aligned.shape}")