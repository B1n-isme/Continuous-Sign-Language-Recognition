import cv2
import numpy as np

# Constants
MAX_HANDS = 2
CROP_SIZE = (112, 112)

def compute_optical_flow(crops):
    """
    Compute Farneback optical flow from RGB crops.
    
    Args:
        crops: np.ndarray of shape (num_frames, num_hands, 112, 112, 3)
    
    Returns:
        flow: np.ndarray of shape (num_frames-1, num_hands, 112, 112, 2)
    """
    num_frames = crops.shape[0]
    flow = np.zeros((num_frames-1, MAX_HANDS, 112, 112, 2), dtype=np.float32)

    for hand_idx in range(MAX_HANDS):
        # Process each hand's sequence
        prev_frame = None
        for frame_idx in range(num_frames):
            # Convert current crop to grayscale
            curr_frame = cv2.cvtColor(crops[frame_idx, hand_idx], cv2.COLOR_RGB2GRAY)
            
            if prev_frame is not None:
                # Compute optical flow between consecutive frames
                flow_frame = cv2.calcOpticalFlowFarneback(
                    prev_frame, curr_frame, None,
                    pyr_scale=0.5,  # Pyramid scale
                    levels=3,      # Number of pyramid levels
                    winsize=15,    # Window size for polynomial expansion
                    iterations=3,  # Iterations per level
                    poly_n=5,      # Polynomial size
                    poly_sigma=1.2,# Gaussian sigma for polynomial
                    flags=0        # Default flags
                )
                flow[frame_idx-1, hand_idx] = flow_frame
            
            prev_frame = curr_frame
    
    return flow

if __name__ == "__main__":
    # Load your .npz file
    data = np.load("data/raw/recording_20250310_201930.npz", allow_pickle=True)
    crops = data["crops"]  # (105, 2, 112, 112, 3)
    skeletal_data = data["skeletal_data"]
    labels = data["labels"]

    # Compute optical flow
    optical_flow = compute_optical_flow(crops)

    # Save updated .npz with optical flow
    np.savez("data/raw/output_with_flow_2.npz",
            skeletal_data=skeletal_data,
            crops=crops,
            optical_flow=optical_flow,
            labels=labels)

    print("Optical flow computed and saved. Shape:", optical_flow.shape)