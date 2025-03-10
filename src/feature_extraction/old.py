import numpy as np
from scipy.interpolate import interp1d
import cv2

# Helper functions (fixed from previous)
def get_detected_mask(skeletal_data):
    T, H, L, C = skeletal_data.shape
    return np.any(skeletal_data != 0, axis=(2, 3))  # (T, H)

# Skeletal preprocessing (unchanged for brevity)
def interpolate_skeletal(skeletal_data):  # ... (as before)
    T, H, L, C = skeletal_data.shape
    interpolated = np.copy(skeletal_data)
    for hand_idx in range(H):
        detected_mask = get_detected_mask(skeletal_data[:, hand_idx:hand_idx+1])[:, 0]
        detected_frames = np.where(detected_mask)[0]
        if len(detected_frames) > 1 and len(detected_frames) < T:
            interp_func = interp1d(detected_frames, skeletal_data[detected_frames, hand_idx],
                                 axis=0, kind='linear', fill_value=0, bounds_error=False)
            interpolated[:, hand_idx] = np.where(detected_mask[:, None, None],
                                                interp_func(np.arange(T)), 0)
    return interpolated

def smooth_skeletal_ema(skeletal_data, alpha=0.3):  # ... (as before)
    T, H, L, C = skeletal_data.shape
    smoothed = np.copy(skeletal_data)
    for hand_idx in range(H):
        detected_mask = get_detected_mask(skeletal_data[:, hand_idx:hand_idx+1])[:, 0]
        for t in range(1, T):
            if detected_mask[t]:
                for l in range(L):
                    for c in range(C):
                        prev_val = smoothed[t-1, hand_idx, l, c] if detected_mask[t-1] else smoothed[t, hand_idx, l, c]
                        smoothed[t, hand_idx, l, c] = (
                            alpha * skeletal_data[t, hand_idx, l, c] +
                            (1 - alpha) * prev_val
                        )
    return smoothed

def rotate_skeletal(skeletal_data, angle_deg):  # ... (as before)
    angle_rad = np.deg2rad(angle_deg)
    rot_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                          [np.sin(angle_rad), np.cos(angle_rad), 0],
                          [0, 0, 1]])
    rotated = np.copy(skeletal_data)
    for t in range(skeletal_data.shape[0]):
        for h in range(skeletal_data.shape[1]):
            if np.any(skeletal_data[t, h] != 0):
                rotated[t, h] = skeletal_data[t, h] @ rot_matrix.T
    return rotated

def time_warp_skeletal(skeletal_data, factor=1.0):  # ... (as before)
    T = skeletal_data.shape[0]
    new_T = int(T * factor)
    old_frames = np.arange(T)
    new_frames = np.linspace(0, T-1, new_T)
    warped = np.zeros((new_T, *skeletal_data.shape[1:]))
    for h in range(2):
        detected_mask = get_detected_mask(skeletal_data[:, h:h+1])[:, 0]
        detected_frames = old_frames[detected_mask]
        if len(detected_frames) > 1:
            for l in range(21):
                for c in range(3):
                    interp = interp1d(detected_frames, skeletal_data[detected_frames, h, l, c],
                                    kind='linear', fill_value=0, bounds_error=False)
                    warped[:, h, l, c] = interp(new_frames)
    return warped

def add_gaussian_noise_skeletal(skeletal_data, std=0.01):  # Renamed for clarity
    T, H, L, C = skeletal_data.shape
    noise = np.random.normal(0, std, skeletal_data.shape)
    detected_mask = get_detected_mask(skeletal_data)[:, :, None, None]  # (T, H, 1, 1)
    return np.where(detected_mask, skeletal_data + noise, skeletal_data)

def frame_drop_skeletal(skeletal_data, drop_rate=0.1):  # ... (as before)
    T = skeletal_data.shape[0]
    keep_mask = np.random.rand(T) > drop_rate
    return skeletal_data[keep_mask]

# Crops preprocessing and augmentation
def normalize_crops(crops):
    return crops.astype(np.float32) / 255.0  # [0, 1]

def rotate_crops(crops, angle_deg):
    T, H, height, width, C = crops.shape
    rotated = np.zeros_like(crops)
    for t in range(T):
        for h in range(H):
            if np.any(crops[t, h] != 0):  # Only rotate non-zero frames
                rotated[t, h] = cv2.warpAffine(crops[t, h], cv2.getRotationMatrix2D((width/2, height/2), angle_deg, 1),
                                              (width, height))
    return rotated

def scale_crops(crops, scale_factor):
    T, H, height, width, C = crops.shape
    new_size = (int(width * scale_factor), int(height * scale_factor))
    scaled = np.zeros((T, H, height, width, C), dtype=crops.dtype)
    for t in range(T):
        for h in range(H):
            if np.any(crops[t, h] != 0):
                resized = cv2.resize(crops[t, h], new_size)
                # Center crop/pad back to 112x112
                start_x = (new_size[0] - width) // 2
                start_y = (new_size[1] - height) // 2
                if scale_factor > 1:
                    scaled[t, h] = resized[start_y:start_y+height, start_x:start_x+width]
                else:
                    scaled[t, h, start_y:start_y+new_size[1], start_x:start_x+new_size[0]] = resized
    return scaled

def occlude_fingers_crops(crops):
    T, H, height, width, C = crops.shape
    occluded = np.copy(crops)
    for t in range(T):
        for h in range(H):
            if np.any(crops[t, h] != 0) and np.random.rand() < 0.5:  # 50% chance
                # Simulate finger occlusion (e.g., top half of hand)
                occluded[t, h, :height//2, :, :] = 0
    return occluded

# Optical flow preprocessing
def normalize_optical_flow(optical_flow):
    magnitude = np.linalg.norm(optical_flow, axis=-1, keepdims=True)  # (T, H, 112, 112, 1)
    max_magnitude = np.max(np.abs(magnitude)) or 1.0  # Avoid div by 0
    return optical_flow / max_magnitude  # [-1, 1]

# Align modalities (updated to include preprocessing)
def align_and_preprocess_modalities(skeletal_variant, crops, optical_flow, variant_name):
    T_new = skeletal_variant.shape[0]
    T_orig = crops.shape[0]
    old_frames = np.arange(T_orig)
    new_frames = np.linspace(0, T_orig-1, T_new)
    
    # Preprocess crops
    crops_normalized = normalize_crops(crops)
    if "rotated_15" in variant_name:
        crops_variant = rotate_crops(crops_normalized, 15)
    elif "rotated_-15" in variant_name:
        crops_variant = rotate_crops(crops_normalized, -15)
    elif "noisy" in variant_name:  # Use occlusion instead of noise for crops
        crops_variant = occlude_fingers_crops(crops_normalized)
    elif "frame_dropped" in variant_name:
        crops_variant = crops_normalized  # Frame drop handled by skeletal
    else:  # original, warped_slow, warped_fast
        crops_variant = crops_normalized
    
    # Interpolate crops
    crops_new = np.zeros((T_new, *crops_variant.shape[1:]), dtype=np.float32)
    for h in range(2):
        for c in range(3):
            interp = interp1d(old_frames, crops_variant[:, h, :, :, c], axis=0)
            crops_new[:, h, :, :, c] = interp(new_frames)
    
    # Preprocess optical flow
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
data = np.load("preprocessed.npz")
skeletal_data = data["skeletal_data"]  # (104, 2, 21, 3) or (105, 2, 21, 3)
crops = data["crops"]
optical_flow = data["optical_flow"]
labels = data["labels"]

if skeletal_data.shape[0] == 105:
    skeletal_data = skeletal_data[:-1]
    crops = crops[:-1]
    optical_flow = optical_flow

# Preprocess and save 7 variants
cleaned_skeletal = smooth_skeletal_ema(interpolate_skeletal(skeletal_data))
variants = [
    ("original", cleaned_skeletal),
    ("rotated_15", rotate_skeletal(cleaned_skeletal, 15)),
    ("rotated_-15", rotate_skeletal(cleaned_skeletal, -15)),
    ("warped_slow", time_warp_skeletal(cleaned_skeletal, 1.2)),
    ("warped_fast", time_warp_skeletal(cleaned_skeletal, 0.8)),
    ("noisy", add_gaussian_noise_skeletal(cleaned_skeletal, 0.01)),
    ("frame_dropped", frame_drop_skeletal(cleaned_skeletal, 0.1))
]

for name, skeletal_variant in variants:
    crops_aligned, flow_aligned = align_and_preprocess_modalities(skeletal_variant, crops, optical_flow, name)
    np.savez(f"video1_{name}.npz",
             skeletal_data=skeletal_variant,
             crops=crops_aligned,
             optical_flow=flow_aligned,
             labels=labels)
    print(f"Saved {name} - Skeletal: {skeletal_variant.shape}, Crops: {crops_aligned.shape}, Flow: {flow_aligned.shape}")