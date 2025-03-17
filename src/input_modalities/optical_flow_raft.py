import cv2
import numpy as np
import torch
from torchvision.models.optical_flow import raft_small

# Constants
MAX_HANDS = 2
TARGET_SIZE = (128, 128)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained RAFT model from torchvision and set to eval mode
model = raft_small(weights=None)
state_dict = torch.load("D:/Data/raft_small_C_T_V2-01064c6d.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()

def preprocess(image):
    """
    Preprocess an image for RAFT: convert to float tensor in [0, 1] with shape (1, 3, H, W).
    """

    # Resize image to TARGET_SIZE (width, height)
    image = cv2.resize(image, TARGET_SIZE)
    # Convert image (H, W, 3) from uint8 to float32 and scale to [0,1]
    image = image.astype(np.float32) / 255.0
    # Convert HWC to CHW format
    image = np.transpose(image, (2, 0, 1))
    # Add batch dimension
    image_tensor = torch.from_numpy(image).unsqueeze(0).to(device)
    return image_tensor

def compute_optical_flow(crops):
    """
    Compute RAFT optical flow from RGB crops.
    
    Args:
        crops: np.ndarray of shape (num_frames, num_hands, 112, 112, 3)
    
    Returns:
        flow: np.ndarray of shape (num_frames-1, num_hands, 112, 112, 2)
    """
    num_frames = crops.shape[0]
    # Initialize output flow array. RAFT flow has 2 channels (u, v).
    flow = np.zeros((num_frames - 1, MAX_HANDS, TARGET_SIZE[0], TARGET_SIZE[1], 2), dtype=np.float32)
    
    for hand_idx in range(MAX_HANDS):
        prev_tensor = None
        for frame_idx in range(num_frames):
            curr_crop = crops[frame_idx, hand_idx]
            curr_tensor = preprocess(curr_crop)  # shape (1, 3, 112, 112)
            
            if prev_tensor is not None:
                # Compute RAFT optical flow between the two consecutive frames.
                # RAFT model returns a list of flow predictions; we use the final prediction.
                list_of_flows = model(prev_tensor, curr_tensor)
                # Select the final (most refined) flow estimate.
                flow_tensor = list_of_flows[-1]  # shape (1, 2, H, W)
                
                # Convert flow tensor to numpy array and reshape from (1, 2, H, W) to (H, W, 2)
                flow_np = flow_tensor[0].detach().permute(1, 2, 0).cpu().numpy()
                
                # Ensure the output flow is exactly TARGET_SIZE (in case of any padding/resizing in RAFT)
                flow[frame_idx - 1, hand_idx] = cv2.resize(flow_np, (TARGET_SIZE[1], TARGET_SIZE[0]))
            
            prev_tensor = curr_tensor
    
    return flow

if __name__ == "__main__":
    # Load your .npz file
    data = np.load("data/raw/Hello-I-Have-Good-Lunch_1dqXkE.npz", allow_pickle=True)
    crops = data["crops"]  # expected shape: (num_frames, 2, 112, 112, 3)
    skeletal_data = data["skeletal_data"]
    labels = data["labels"]

    # Compute optical flow using RAFT
    optical_flow = compute_optical_flow(crops)

    # Save updated .npz with optical flow
    np.savez("data/raw/output_with_flow_2.npz",
             skeletal_data=skeletal_data,
             crops=crops,
             optical_flow=optical_flow,
             labels=labels)

    print("Optical flow computed and saved. Shape:", optical_flow.shape)
