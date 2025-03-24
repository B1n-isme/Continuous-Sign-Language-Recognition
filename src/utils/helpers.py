import cv2
import numpy as np

def to_base64(num):
    """
    Convert a number to base64 encoding using only filename-safe characters.

    Args:
        num (int): The number to convert

    Returns:
        str: Base64 encoded string safe for filenames
    """
    # Use a filename-safe character set (no '/' or '+' that might be in standard base64)
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
    if num == 0:
        return "0"

    result = ""
    while num > 0:
        result = chars[num % 64] + result
        num //= 64

    return result


def get_bounding_box(landmarks, width, height, margin=20):
    """
    Compute bounding box from hand landmarks with adjustable margin.

    Args:
        landmarks: MediaPipe hand landmarks object.
        width: Frame width.
        height: Frame height.
        margin: Pixel margin to add around the detected hand.

    Returns:
        tuple: (x_min, y_min, x_max, y_max) coordinates.
    """
    # Vectorized approach for better performance
    x_coords = np.array([lm.x for lm in landmarks.landmark]) * width
    y_coords = np.array([lm.y for lm in landmarks.landmark]) * height

    x_min = max(0, int(np.min(x_coords)) - margin)
    y_min = max(0, int(np.min(y_coords)) - margin)
    x_max = min(width, int(np.max(x_coords)) + margin)
    y_max = min(height, int(np.max(y_coords)) + margin)

    return x_min, y_min, x_max, y_max


def resize_preserve_aspect_ratio(
    image, 
    target_size,
    padding_color=(0, 0, 0),  # Configurable padding (BGR format)
    interpolation=cv2.INTER_LINEAR  # Better for upscaling
):
    """
    Resize BGR image while preserving aspect ratio, padding with specified color.

    Args:
        image: Input BGR image (numpy array, 3 channels)
        target_size: Desired output size (width, height)
        padding_color: BGR tuple for padding (default: black)
        interpolation: Interpolation method for resizing

    Returns:
        Resized and padded BGR image
    """
    target_width, target_height = target_size
    height, width = image.shape[:2]

    # Calculate scaling ratio
    width_ratio = target_width / width
    height_ratio = target_height / height
    ratio = min(width_ratio, height_ratio)

    # Compute new dimensions
    new_width = int(width * ratio)
    new_height = int(height * ratio)

    # Resize with appropriate interpolation
    resized = cv2.resize(
        image, 
        (new_width, new_height), 
        interpolation=interpolation if ratio < 1 else cv2.INTER_LINEAR
    )

    # Create a canvas with the target size and padding color
    canvas = np.full((target_height, target_width, 3), padding_color, dtype=np.uint8)

    # Calculate padding offsets (centered)
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    # Place resized image on canvas
    canvas[
        y_offset : y_offset + new_height,
        x_offset : x_offset + new_width
    ] = resized

    return canvas


