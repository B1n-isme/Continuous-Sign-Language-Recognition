# skeletal_aug.py
import numpy as np
from scipy.interpolate import interp1d

def get_detected_mask(skeletal_data):
    """Returns (T, H) mask indicating detected hands."""
    T, H, L, C = skeletal_data.shape
    return np.any(skeletal_data != 0, axis=(2, 3))

def interpolate_skeletal(skeletal_data):
    """Interpolates missing frames for detected hands, preserving zeros."""
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

def smooth_skeletal_ema(skeletal_data, alpha=0.3):
    """Applies EMA smoothing to detected hands."""
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

def scale_skeletal(skeletal_data, scale_factor):
    """Scales skeletal coordinates by scale_factor."""
    scaled = np.copy(skeletal_data) * scale_factor
    return scaled

def rotate_skeletal(skeletal_data, angle_deg=None):
    """Rotates detected hands by angle_deg (random if None)."""
    if angle_deg is None:
        angle_deg = np.random.uniform(-15, 15)
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

def time_warp_skeletal(skeletal_data, factor=1.0):
    """Warps time by factor, interpolating detected hands."""
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

def add_gaussian_noise_skeletal(skeletal_data, std=0.01):
    """Adds Gaussian noise to detected hands."""
    T, H, L, C = skeletal_data.shape
    noise = np.random.normal(0, std, skeletal_data.shape)
    detected_mask = get_detected_mask(skeletal_data)[:, :, None, None]
    return np.where(detected_mask, skeletal_data + noise, skeletal_data)

def frame_drop_skeletal(skeletal_data, drop_rate=0.1):
    """Randomly drops frames."""
    T = skeletal_data.shape[0]
    keep_mask = np.random.rand(T) > drop_rate
    return skeletal_data[keep_mask]