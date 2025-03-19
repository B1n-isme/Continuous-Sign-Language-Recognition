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


def resize_preserve_aspect_ratio(image, target_size):
    """
    Resize image while preserving aspect ratio, then pad to target size.

    Args:
        image: Input image (numpy array)
        target_size: Desired output size as (width, height)

    Returns:
        Resized and padded image
    """
    target_width, target_height = target_size
    height, width = image.shape[:2]

    # Calculate the ratio of the target dimensions to the original dimensions
    width_ratio = target_width / width
    height_ratio = target_height / height

    # Use the smaller ratio to ensure the image fits within the target dimensions
    ratio = min(width_ratio, height_ratio)

    # Calculate new dimensions
    new_width = int(width * ratio)
    new_height = int(height * ratio)

    # Resize the image
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a black canvas of the target size
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # Calculate offsets to center the image
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    # Place the resized image on the canvas
    canvas[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = resized

    return canvas


