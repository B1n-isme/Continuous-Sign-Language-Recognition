import cv2
import time
import urllib.request
import mediapipe as mp
import numpy as np

# Global list to collect landmark results per frame.
collected_landmarks = []

# Download the pre-trained hand landmark model.
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/hand_landmark.task"
model_filename = "hand_landmarker.task"
urllib.request.urlretrieve(MODEL_URL, model_filename)
model_path_full = model_filename

# Import MediaPipe task classes.
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


def print_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    
    if result.hand_landmarks and result.handedness:
        # Iterate over each detected hand.
        for hand_index, (landmarks, handedness_list) in enumerate(zip(result.hand_landmarks, result.handedness)):
            # Extract hand type from the first classification entry.
            hand_type = handedness_list[0].category_name  # Expected 'Left' or 'Right'
            print(f"Hand {hand_index} type: {hand_type}")
            # Iterate over each of the 21 landmarks.
            for idx, landmark in enumerate(landmarks):
                print(f"Landmark {idx}: x={landmark.x}, y={landmark.y}, z={landmark.z}")



def main():
    global collected_landmarks
    # Initialize webcam capture.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Create HandLandmarkerOptions using the downloaded model.
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path_full),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result,
    )

    # Create the HandLandmarker instance.
    with HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Mirror the frame for a selfie view.
            frame = cv2.flip(frame, 1)

            # Convert frame from BGR (OpenCV) to RGB.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Get the current timestamp in milliseconds.
            timestamp_ms = int(time.time() * 1000)

            # Process the frame asynchronously.
            landmarker.detect_async(mp_image, timestamp_ms)

            # Display the mirrored frame.
            cv2.imshow("Hand Landmarker", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Release resources.
    cap.release()
    cv2.destroyAllWindows()

    # Convert the collected results into a NumPy array of shape (num_frames, 42, 3).
    if collected_landmarks:
        all_landmarks_array = np.array(collected_landmarks)
        print("Collected landmarks array shape:", all_landmarks_array.shape)
        # Save the collected data to disk.
        np.save("collected_landmarks.npy", all_landmarks_array)
    else:
        print("No landmarks collected.")


if __name__ == "__main__":
    main()
