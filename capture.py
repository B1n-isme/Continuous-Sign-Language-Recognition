import cv2
import mediapipe as mp
import time
import argparse
from datetime import datetime

# Set up argument parser.
parser = argparse.ArgumentParser(description="Auto Video Recorder on Hand Detection")
parser.add_argument("--word", type=str, required=True,
                    help="A word representing the recorded action (used in the filename)")
args = parser.parse_args()

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Open the default camera (selfie camera).
cap = cv2.VideoCapture(0)

# Parameters.
recording = False         # True when recording is in progress.
record_duration = 1.0     # Duration to record (in seconds).
record_start_time = None
video_writer = None
fps = 30                  # Adjust as needed based on your camera.
# Flag to ensure we wait for the hand to be removed before starting a new session.
waiting_for_hand_release = False

print("Starting hand detection. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for mirror view and convert to RGB for MediaPipe.
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame for hand detection.
    results = hands.process(frame_rgb)
    hand_detected = results.multi_hand_landmarks is not None

    if not recording:
        # Only trigger a new recording if we're not waiting for hand release.
        if hand_detected and not waiting_for_hand_release:
            # Start recording.
            recording = True
            record_start_time = time.time()

            # Generate filename using the input word and current timestamp.
            timestamp = int(time.time())  # Get current Unix timestamp
            filename = f"{timestamp}_{args.word}.avi"
            print(f"Recording session started: {filename}")

            # Prepare VideoWriter to record video.
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            height, width, _ = frame.shape
            video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        elif not hand_detected:
            # Once no hand is detected, we are ready to trigger a new recording.
            waiting_for_hand_release = False
        else:
            # If hand is detected but we're already waiting for it to go away, do nothing.
            waiting_for_hand_release = True
    else:
        # During recording, write the frame to the video file.
        video_writer.write(frame)
        if time.time() - record_start_time >= record_duration:
            # End of recording session.
            recording = False
            waiting_for_hand_release = True  # Start waiting for the hand to be removed.
            video_writer.release()
            video_writer = None
            print("Recording session ended.")

    cv2.imshow("Auto Video Recording", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
