import numpy as np
import cv2
from scipy.interpolate import interp1d

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
    """Scales crops by scale_factor, resizing back to original dimensions."""
    T, H, height, width, C = crops.shape
    new_size = (int(width * scale_factor), int(height * scale_factor))
    scaled = np.zeros((T, H, height, width, C), dtype=crops.dtype)
    for t in range(T):
        for h in range(H):
            if np.any(crops[t, h] != 0):
                resized = cv2.resize(crops[t, h], new_size, interpolation=cv2.INTER_AREA)
                if scale_factor > 1:
                    # Crop center for larger scales
                    start_x = (new_size[0] - width) // 2
                    start_y = (new_size[1] - height) // 2
                    scaled[t, h] = resized[start_y:start_y+height, start_x:start_x+width]
                else:
                    # Pad center for smaller scales
                    pad_x = (width - new_size[0]) // 2
                    pad_y = (height - new_size[1]) // 2
                    scaled[t, h, pad_y:pad_y+new_size[1], pad_x:pad_x+new_size[0]] = resized
    return scaled

def occlude_fingers_crops(crops):
    """Simulates finger occlusion on 50% of non-zero frames."""
    T, H, height, width, C = crops.shape
    occluded = np.copy(crops)
    for t in range(T):
        for h in range(H):
            if np.any(crops[t, h] != 0) and np.random.rand() < 0.5:  # 50% chance
                # Simulate finger occlusion (e.g., top half of hand)
                occluded[t, h, :height//2, :, :] = 0
    return occluded