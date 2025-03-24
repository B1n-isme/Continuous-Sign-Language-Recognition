import numpy as np
import cv2

def normalize_crops(crops):
    """
    Converts image crops to float32 and normalizes pixel values to [0, 1].
    """
    return crops.astype(np.float32) / 255.0

def rotate_crops(crops, angle_deg):
    """
    Rotates each crop by the specified angle (in degrees) around its center.
    Only non-zero crops (i.e. with some hand content) are rotated.
    """
    if crops.ndim != 5:
        raise ValueError("Expected crops to be a 5D array (T, H, height, width, C)")
    T, H, height, width, C = crops.shape
    rotated = np.zeros_like(crops)
    center = (width / 2, height / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1)
    for t in range(T):
        for h in range(H):
            crop = crops[t, h]
            # Process only if crop contains any non-zero value.
            if np.any(crop != 0):
                rotated[t, h] = cv2.warpAffine(crop, rot_mat, (width, height),
                                               flags=cv2.INTER_LINEAR,
                                               borderMode=cv2.BORDER_CONSTANT,
                                               borderValue=0)
    return rotated

def scale_crops(crops, scale_factor):
    """
    Scales each crop by scale_factor and then resizes back to the original dimensions.
    For scale_factor > 1, the resized image is cropped at the center.
    For scale_factor < 1, the resized image is padded equally on all sides.
    """
    if crops.ndim != 5:
        raise ValueError("Expected crops to be a 5D array (T, H, height, width, C)")
    T, H, height, width, C = crops.shape
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    new_size = (new_width, new_height)
    scaled = np.zeros((T, H, height, width, C), dtype=crops.dtype)
    
    for t in range(T):
        for h in range(H):
            crop = crops[t, h]
            if np.any(crop != 0):
                # Resize the crop.
                resized = cv2.resize(crop, new_size, interpolation=cv2.INTER_AREA)
                if scale_factor > 1:
                    # For larger images, crop the center.
                    start_x = (new_width - width) // 2
                    start_y = (new_height - height) // 2
                    scaled[t, h] = resized[start_y:start_y+height, start_x:start_x+width]
                else:
                    # For smaller images, pad evenly.
                    pad_x = (width - new_width) // 2
                    pad_y = (height - new_height) // 2
                    # Place resized image in the center of a black canvas.
                    scaled[t, h, pad_y:pad_y+new_height, pad_x:pad_x+new_width] = resized
    return scaled

def occlude_fingers_crops(crops):
    """
    Simulates finger occlusion on 50% of non-zero crops.
    Currently occludes the top half of the crop.
    You might later consider randomizing the occlusion region or shape for more diversity.
    """
    if crops.ndim != 5:
        raise ValueError("Expected crops to be a 5D array (T, H, height, width, C)")
    T, H, height, width, C = crops.shape
    occluded = np.copy(crops)
    for t in range(T):
        for h in range(H):
            crop = crops[t, h]
            if np.any(crop != 0) and np.random.rand() < 0.5:
                # Occlude top half; you could modify this to a random region if desired.
                occluded[t, h, :height//2, :, :] = 0
    return occluded
