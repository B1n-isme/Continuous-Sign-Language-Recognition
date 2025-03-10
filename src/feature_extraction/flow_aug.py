import numpy as np
from scipy.interpolate import interp1d

# Optical flow preprocessing
def normalize_optical_flow(optical_flow):
    magnitude = np.linalg.norm(optical_flow, axis=-1, keepdims=True)  # (T, H, 112, 112, 1)
    max_magnitude = np.max(np.abs(magnitude)) or 1.0  # Avoid div by 0
    return optical_flow / max_magnitude  # [-1, 1]