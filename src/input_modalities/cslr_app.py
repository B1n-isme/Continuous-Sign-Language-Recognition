import streamlit as st
import cv2
import mediapipe as mp
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import threading
import time
from datetime import datetime
import sys
import tempfile
import shutil
from input_modalities.optical_flow_raft import compute_optical_flow

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.config import load_config
from src.utils.video_processing import VideoProcessor  # Import the moved components

# Load configuration
CONFIG_PATH = "configs/data_config.yaml"
config = load_config(CONFIG_PATH)
DATA_DIR = config["paths"]["raw_data_root"]
CROP_SIZE = tuple(config["extraction"]["crop_resolution"])
MAX_HANDS = config["extraction"]["max_hands"]
RESIZE_METHOD = config["extraction"].get("resize_method", "fixed")

# Ensure the data directory exists
os.makedirs(DATA_DIR, exist_ok=True)


# Create a clean temporary directory for streamlit media files
def setup_temp_directory():
    """
    Setup a clean temporary directory for Streamlit media files.

    Returns:
        str: Path to temporary directory
    """
    temp_dir = os.path.join(tempfile.gettempdir(), "streamlit_cslr_app")
    if os.path.exists(temp_dir):
        try:
            # Clean up existing temp files
            shutil.rmtree(temp_dir)
        except:
            pass
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir


def to_base62(num):
    """
    Convert a number to base62 encoding.

    Args:
        num (int): The number to convert

    Returns:
        str: Base62 encoded string
    """
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    if num == 0:
        return "0"

    result = ""
    while num > 0:
        result = chars[num % 62] + result
        num //= 62

    return result


def save_recording_data(video_processor, label_glosses=None):
    """
    Save recorded data to a file.

    Args:
        video_processor: VideoProcessor instance with recorded data
        label_glosses: List of glosses for this recording

    Returns:
        Path to the saved file or None if no data was saved
    """
    # Check if there's data to save
    if not video_processor.skeletal_data:
        return None

    # Convert skeletal data to numpy array
    skeletal_array = np.array(video_processor.skeletal_data)

    # Convert crops array
    crops_array = np.array(video_processor.crops_data)

    optical = compute_optical_flow(crops_array)

    # Generate output filename using Unix timestamp encoded in base62
    unix_timestamp = int(time.time())  # Get current Unix timestamp (seconds)
    timestamp = to_base62(
        unix_timestamp
    )  # Convert to base62 for shorter representation

    # Create glosses label part
    if label_glosses and len(label_glosses) > 0:
        gloss_label = "-".join(label_glosses)
    else:
        gloss_label = "Unlabeled"

    output_file = os.path.join(DATA_DIR, f"{gloss_label}_{timestamp}.npz")

    # Save the data
    try:
        np.savez_compressed(
            output_file,
            skeletal_data=skeletal_array,
            crops=crops_array,
            optical_flow=optical,
            labels=np.array(label_glosses) if label_glosses else np.array([]),
        )
        return output_file
    except Exception as e:
        st.error(f"Error saving data: {e}")
        return None


def app():
    # Set up temporary directory
    temp_dir = setup_temp_directory()

    # Initialize session state
    if "processor" not in st.session_state:
        # Create VideoProcessor with configuration from config file
        st.session_state.processor = VideoProcessor(
            max_hands=MAX_HANDS, crop_size=CROP_SIZE, resize_method=RESIZE_METHOD
        )
        st.session_state.camera_started = False
        st.session_state.last_recording_status = None
        st.session_state.frame_counter = 0  # Track frames for display rate limiting

    # Initialize glosses_clear_requested flag if needed
    if "glosses_clear_requested" not in st.session_state:
        st.session_state.glosses_clear_requested = False

    # Handle glosses clearing if requested
    if st.session_state.glosses_clear_requested:
        # Reset the flag first
        st.session_state.glosses_clear_requested = False
        # The input will be empty on the next run

    # Create placeholder for the video feed
    video_placeholder = st.empty()
    status_text = st.empty()

    # Input for gloss labels - put above the video for better visibility
    glosses_input = st.text_input(
        "Enter glosses (words) for this recording, separated by spaces:",
        key="glosses_input",
        value="" if st.session_state.get("glosses_clear_requested", False) else None,
    )

    # Parse glosses
    glosses = [g.strip() for g in glosses_input.split()] if glosses_input else []

    if glosses:
        st.write("**Detected glosses:**", ", ".join(glosses))

    # Clear button for glosses - modify this to use the flag approach
    if st.button("Clear Glosses"):
        st.session_state.glosses_clear_requested = True
        st.rerun()

    # Start camera automatically if not already started
    if not st.session_state.camera_started:
        if st.session_state.processor.start():
            st.session_state.camera_started = True

    # Update the status message
    if st.session_state.camera_started:
        # Main video frame update loop
        try:
            while True:
                frame = st.session_state.processor.get_latest_frame()
                if frame is not None:
                    try:
                        # Limit frame rate for display - update every 3rd frame
                        st.session_state.frame_counter += 1
                        if st.session_state.frame_counter % 3 == 0:
                            # Save frame as an image file to avoid memory issues
                            frame_path = os.path.join(temp_dir, "current_frame.jpg")
                            cv2.imwrite(frame_path, frame)

                            # Display from file rather than from memory
                            if os.path.exists(frame_path):
                                video_placeholder.image(
                                    frame_path, channels="BGR", use_container_width=True
                                )
                    except Exception as display_err:
                        # Just log the error and continue - don't crash on display errors
                        st.error(f"Display error (continuing): {display_err}")
                        time.sleep(0.1)  # Brief pause to avoid flooding with errors

                # Check for auto-save pending (recording just ended due to no hands)
                if getattr(st.session_state.processor, "auto_save_pending", False):
                    # Save the recording that just ended automatically
                    recording_data = st.session_state.processor.stop_recording()
                    if recording_data["frames"] > 0:
                        saved_file = save_recording_data(
                            st.session_state.processor, glosses
                        )
                        if saved_file:
                            status_text.success(
                                f"Recording automatically stopped and saved to: {saved_file} ({recording_data['frames']} frames)"
                            )
                    # Reset the flag
                    st.session_state.processor.auto_save_pending = False
                    continue  # Skip the rest of this iteration

                # Check if we should auto-record based on glosses and hands
                recording_status = st.session_state.processor.auto_record_if_needed(
                    bool(glosses), no_hands_threshold=30
                )

                # Handle status changes
                if (
                    recording_status["started"]
                    and recording_status != st.session_state.last_recording_status
                ):
                    status_text.success(
                        "Recording started automatically - hand detected!"
                    )

                st.session_state.last_recording_status = recording_status

                # Display ongoing status message
                if st.session_state.processor.is_recording:
                    status_text.info(
                        f"Recording... Frames: {st.session_state.processor.frame_counter}"
                    )
                elif not glosses:
                    status_text.info(
                        "Enter glosses to enable automatic recording when hands are shown."
                    )
                else:
                    status_text.info(
                        "Camera active. Show your hands to start recording."
                    )

                # Add a small sleep to prevent high CPU usage and resource exhaustion
                time.sleep(0.03)  # ~33 fps cap

        except KeyboardInterrupt:
            st.warning("User interrupted the application.")
        except Exception as e:
            st.error(f"Error in video processing: {str(e)}")
            import traceback

            st.code(traceback.format_exc())

        # Ensure camera stops when app is closed
        st.session_state.on_script_end = st.session_state.processor.stop

    # Add a button to manually clear cache and restart the app if needed
    if st.button("Clear Cache and Restart"):
        st.cache_data.clear()
        st.session_state.clear()
        st.rerun()


if __name__ == "__main__":
    app()
