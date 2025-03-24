# skeletal_aug.py
import numpy as np
from scipy.interpolate import interp1d

def get_detected_mask(skeletal_data):
    """
    Returns a Boolean mask of shape (T, H) indicating detected hands.
    A frame is considered detected if any landmark coordinate is nonzero.
    """
    T, H, L, C = skeletal_data.shape
    return np.any(skeletal_data != 0, axis=(2, 3))

def interpolate_skeletal(skeletal_data):
    """
    Interpolates missing frames for each hand that has at least two detected frames.
    Only frames where a detection exists (i.e. nonzero) are updated.
    Hands with all-zero data remain untouched.
    """
    T, H, L, C = skeletal_data.shape
    interpolated = np.copy(skeletal_data)
    for hand in range(H):
        hand_data = skeletal_data[:, hand]  # shape: (T, L, C)
        detected_mask = get_detected_mask(skeletal_data[:, hand:hand+1])[:, 0]  # shape: (T,)
        detected_frames = np.where(detected_mask)[0]
        # Only interpolate if there are at least two detected frames.
        if len(detected_frames) > 1:
            interp_func = interp1d(detected_frames, hand_data[detected_frames],
                                   axis=0, kind='linear', fill_value=0, bounds_error=False)
            interpolated_hand = interp_func(np.arange(T))
            # Only update frames that originally had detection; leave the zero frames as is.
            for t in range(T):
                if detected_mask[t]:
                    interpolated[t, hand] = interpolated_hand[t]
    return interpolated

def smooth_skeletal_ema(skeletal_data, alpha=0.3):
    """
    Applies Exponential Moving Average (EMA) smoothing along the temporal axis
    for each hand, but only on frames that are detected.
    If a previous frame is missing, the current raw value is used.
    """
    T, H, L, C = skeletal_data.shape
    smoothed = np.copy(skeletal_data)
    for hand in range(H):
        detected_mask = get_detected_mask(skeletal_data[:, hand:hand+1])[:, 0]
        for t in range(1, T):
            if detected_mask[t]:
                # Use the previous smoothed value if the previous frame was detected,
                # otherwise fallback to the current raw value.
                prev_val = smoothed[t-1, hand] if detected_mask[t-1] else skeletal_data[t, hand]
                # Vectorized over landmarks and channels.
                smoothed[t, hand] = alpha * skeletal_data[t, hand] + (1 - alpha) * prev_val
    return smoothed

def scale_skeletal(skeletal_data, scale_factor):
    """
    Scales skeletal coordinates by the provided scale factor.
    """
    return np.copy(skeletal_data) * scale_factor

def rotate_skeletal(skeletal_data, angle_deg=None):
    """
    Rotates detected hands by a given angle in degrees.
    If no angle is provided, a random angle between -15 and 15 degrees is used.
    Only hands with nonzero values are rotated.
    """
    if angle_deg is None:
        angle_deg = np.random.uniform(-15, 15)
    angle_rad = np.deg2rad(angle_deg)
    # 3D rotation matrix (assuming the last coordinate is used, e.g., [x, y, z])
    rot_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                           [np.sin(angle_rad),  np.cos(angle_rad), 0],
                           [0, 0, 1]])
    T, H, L, C = skeletal_data.shape
    rotated = np.copy(skeletal_data)
    for t in range(T):
        for h in range(H):
            if np.any(skeletal_data[t, h] != 0):
                # Apply rotation; note that matrix multiplication here assumes each landmark is 3D.
                rotated[t, h] = skeletal_data[t, h] @ rot_matrix.T
    return rotated

def time_warp_skeletal(skeletal_data, factor=1.0):
    """
    Applies time warping by a given factor.
    For each hand, detected frames are interpolated to form a new sequence length.
    If only one detection exists, the value is simply replicated.
    Hands with no detections remain zeros.
    """
    T, H, L, C = skeletal_data.shape
    new_T = int(T * factor)
    old_frames = np.arange(T)
    new_frames = np.linspace(0, T-1, new_T)
    warped = np.zeros((new_T, H, L, C))
    for h in range(H):
        hand_data = skeletal_data[:, h]
        detected_mask = get_detected_mask(skeletal_data[:, h:h+1])[:, 0]
        detected_frames = old_frames[detected_mask]
        if len(detected_frames) > 1:
            for l in range(L):
                for c in range(C):
                    interp = interp1d(detected_frames, hand_data[detected_mask, l, c],
                                      kind='linear', fill_value=0, bounds_error=False)
                    warped[:, h, l, c] = interp(new_frames)
        elif len(detected_frames) == 1:
            # If only one detected frame exists, replicate its value across the warped sequence.
            warped[:, h] = hand_data[detected_frames[0]]
        # Else: no detection -> remains zeros.
    return warped

def add_gaussian_noise_skeletal(skeletal_data, std=0.01):
    """
    Adds Gaussian noise to the detected hand coordinates.
    Noise is only added to frames where a detection is present.
    """
    noise = np.random.normal(0, std, skeletal_data.shape)
    detected_mask = get_detected_mask(skeletal_data)[:, :, None, None]
    return np.where(detected_mask, skeletal_data + noise, skeletal_data)

def frame_drop_skeletal(skeletal_data, drop_rate=0.1):
    """
    Randomly drops frames with a given drop rate.
    Downstream processes must handle the variable sequence length.
    """
    T = skeletal_data.shape[0]
    keep_mask = np.random.rand(T) > drop_rate
    return skeletal_data[keep_mask]

def shear_skeletal(skeletal_data, shear_x=None, shear_y=None):
    """
    Applies a shear transformation (affine transform) along the x and y axes
    to the detected hands for additional spatial augmentation.
    If shear_x or shear_y is not provided, a random shear factor in the range
    [-0.2, 0.2] is used.
    """
    T, H, L, C = skeletal_data.shape
    if shear_x is None:
        shear_x = np.random.uniform(-0.2, 0.2)
    if shear_y is None:
        shear_y = np.random.uniform(-0.2, 0.2)
    # Construct a shear matrix for 2D transformation (applied to [x, y]; z remains unchanged)
    shear_matrix = np.array([[1, shear_x, 0],
                             [shear_y, 1, 0],
                             [0, 0, 1]])
    transformed = np.copy(skeletal_data)
    for t in range(T):
        for h in range(H):
            if np.any(skeletal_data[t, h] != 0):
                transformed[t, h] = skeletal_data[t, h] @ shear_matrix.T
    return transformed
