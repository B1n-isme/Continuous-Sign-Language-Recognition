import numpy as np

# Optical flow preprocessing
def normalize_optical_flow(optical_flow):
    magnitude = np.linalg.norm(optical_flow, axis=-1, keepdims=True)  # (T, H, 112, 112, 1)
    # Compute global maximum magnitude. Since magnitude is non-negative, no need for abs.
    max_magnitude = np.max(magnitude)
    
    # Avoid division by zero: if max_magnitude is too small, set it to a small epsilon.
    eps = 1e-6
    max_magnitude = max_magnitude if max_magnitude > eps else 1.0    
    
    return optical_flow / max_magnitude  # [-1, 1]