import cv2
import mediapipe as mp
import time
import argparse
from datetime import datetime
import threading
import queue

# ----------------- Argument Parsing -----------------
parser = argparse.ArgumentParser(description="Auto Video Recorder on Hand Detection")
parser.add_argument("--word", type=str, required=True,
                    help="A word representing the recorded action (used in the filename)")
args = parser.parse_args()

# ----------------- Initialize MediaPipe Hands -----------------
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    min_detection_confidence=0.7, 
    max_num_hands=2,
    min_tracking_confidence=0.7)

# ----------------- Open Camera -----------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# ----------------- Global Parameters & Flags -----------------
fps = 30                   # Desired FPS for recording.
record_duration = 1.0      # Recording length in seconds.
recording = False          # True when recording is in progress.
is_waiting_for_hand_release = False
video_writer = None
record_start_time = None

# Queues for decoupling frame capture and processing.
frame_queue = queue.Queue(maxsize=10)
# latest_frame is used for display, protected by a lock.
latest_frame = None
display_frame_lock = threading.Lock()

# Flag to signal threads to stop.
stop_threads = False

# ----------------- Capture Thread -----------------
def capture_frames():
    global stop_threads
    while not stop_threads:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from camera.")
            continue
        # Flip frame for a mirror view.
        frame = cv2.flip(frame, 1)
        try:
            frame_queue.put(frame, timeout=0.01)
        except queue.Full:
            # Skip frame if queue is full.
            continue

# ----------------- Processing (Detection & Recording) Thread -----------------
def process_frames():
    global recording, is_waiting_for_hand_release, video_writer, record_start_time, latest_frame, stop_threads
    while not stop_threads:
        try:
            frame = frame_queue.get(timeout=0.05)
        except queue.Empty:
            continue
        
        # Update global latest_frame for display.
        with display_frame_lock:
            latest_frame = frame.copy()
        
        if not recording:
            # Perform hand detection only if not recording.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_detector.process(frame_rgb)
            is_hand = results.multi_hand_landmarks is not None
            
            if is_hand and not is_waiting_for_hand_release:
                # Hand detected: start recording.
                recording = True
                record_start_time = time.time()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Using mp4 container with H264 codec for higher quality.
                filename = f"{args.word}_{timestamp}.mp4"
                print(f"Recording session started: {filename}")
                fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H264 codec (ensure your OpenCV build supports it).
                height, width = frame.shape[:2]
                video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
                if not video_writer.isOpened():
                    print("Error: VideoWriter failed to open.")
                    recording = False
                    continue
                # Write the current frame.
                video_writer.write(frame)
            elif not is_hand:
                # If no hand is detected, clear waiting flag.
                is_waiting_for_hand_release = False
            else:
                # Hand still present – keep waiting.
                is_waiting_for_hand_release = True
        else:
            # When recording, write each frame to the video.
            if video_writer is not None:
                video_writer.write(frame)
            # Check if the recording duration has been met.
            if time.time() - record_start_time >= record_duration:
                recording = False
                is_waiting_for_hand_release = True  # Wait for the hand to be removed.
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                print("Recording session ended.")
        
        frame_queue.task_done()

# ----------------- Start Threads -----------------
capture_thread = threading.Thread(target=capture_frames, daemon=True)
processing_thread = threading.Thread(target=process_frames, daemon=True)

capture_thread.start()
processing_thread.start()

print("Starting hand detection. Press 'q' to exit.")

# ----------------- Main Loop: Display Frames -----------------
while True:
    with display_frame_lock:
        if latest_frame is not None:
            # Resize the frame to 80% of its original resolution for display.
            disp_frame = cv2.resize(latest_frame, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)
        else:
            continue
    
    cv2.imshow("Auto Video Recording", disp_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop_threads = True
        break

# ----------------- Cleanup -----------------
capture_thread.join(timeout=1.0)
processing_thread.join(timeout=1.0)
cap.release()
cv2.destroyAllWindows()
hands_detector.close()
