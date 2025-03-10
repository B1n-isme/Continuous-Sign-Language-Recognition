import numpy as np
import matplotlib.pyplot as plt

# Load the .npz file
data = np.load("data/raw/output_with_flow.npz", allow_pickle=True)
skeletal_data = data['skeletal_data']  # Shape: (105, 2, 21, 3)

# Number of frames to visualize
num_frames = len(skeletal_data)

# Define connections between hand landmarks (MediaPipe Hand landmark connections)
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),   # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),   # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]

# Sequentially plot hand landmarks
fig = plt.figure(figsize=(5, 5))

for i in range(num_frames):
    plt.clf()  # Clear previous frame
    
    for hand_idx in range(2):  # Left & Right hands
        hand_landmarks = skeletal_data[i, hand_idx]  # Shape: (21, 3)
        
        if np.all(hand_landmarks == 0):  # Skip empty hands
            continue

        x, y, z = hand_landmarks[:, 0], hand_landmarks[:, 1], hand_landmarks[:, 2]

        # Plot landmarks
        plt.scatter(x, -y, s=20, color='blue' if hand_idx == 0 else 'red', label=f"Hand {hand_idx+1}")

        # Plot connections
        for (start, end) in connections:
            plt.plot([x[start], x[end]], [-y[start], -y[end]], color='black')

    plt.title(f"Frame {i+1}")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.axis("off")
    plt.pause(0.1)  # Pause to create an animation effect

plt.show()
