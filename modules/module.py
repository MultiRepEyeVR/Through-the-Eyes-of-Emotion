import json
import sys
import cv2
import numpy as np
import gc
import pandas as pd
import math


def load_config_file(config_file_path):
    with open(config_file_path, 'r') as file:
        config = json.load(file)

    print(config)

    return config


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def get_elicitation_time_by_subject(timeline, session, subject):
    if session in timeline:
        session_info = timeline[session]

        for key, value_list in session_info.items():
            if subject in value_list:
                return key
            
    return "none"


def load_frames(video_path, window_time, start_time, fps, sample_rate, preprocess='none', resize=(224, 224), frame_channels=1, overlap=0.0):
    # Check if overlap is a valid value
    if not 0.0 <= overlap < 1.0:
        sys.exit("Error: Invalid overlap value. Expected a float in the range [0.0, 1.0).")

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        sys.exit(f"Error: Could not open video {video_path}.")

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    video_duration = total_frames / fps

    end_frame = max(0, int(int(fps * start_time) - window_time * fps))

    frame_skip = int(fps / sample_rate)

    frames_list = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames_list.append(frame)

    video.release()

    frames_array = []
    for i in range(len(frames_list) - 1, end_frame - 1, -frame_skip):
        frame = frames_list[i]
        if preprocess == 'flip_horizontal':
            frame = cv2.flip(frame, 0)
        elif preprocess == 'flip_vertical':
            frame = cv2.flip(frame, 1)
        elif preprocess == 'flip_both':
            frame = cv2.flip(frame, -1)
        if resize:
            frame = cv2.resize(frame, resize)
        if frame_channels == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = frame[:, :, np.newaxis]
        frames_array.append(frame)

    frames_array = np.array(frames_array)

    del frames_list
    gc.collect()

    number_frames_in_window = int(window_time * sample_rate)

    total_frames = frames_array.shape[0]

    overlap_frames = int(number_frames_in_window * overlap)

    stride = number_frames_in_window - overlap_frames

    number_batches = (total_frames - overlap_frames) // stride

    overlapping_batches = []

    for i in range(number_batches):
        start_index = i * stride
        end_index = start_index + number_frames_in_window
        batch = frames_array[start_index:end_index]
        overlapping_batches.append(batch)

    # Shape: [number_batches, number_frames_in_window, height, width, channels]
    frames_array = np.array(overlapping_batches, dtype=np.float32)

    frames_array = np.flip(frames_array, axis=1)  # Inverse the order of frames in each batch

    # Reshape: [number_batches, channels, frames_per_batch, height, width]
    frames_array = frames_array.transpose(0, 4, 1, 2, 3)

    return frames_array, video_duration


def load_gaze_and_diameter(gaze_df, pupil_df, window_time, fps, duration, overlap=0.0):
    target_rows = int(math.ceil(duration) * fps)

    # Sample gaze data with target rows
    gaze_indices = np.linspace(0, len(gaze_df) - 1, num=target_rows, dtype=int)
    sampled_gaze = gaze_df.iloc[gaze_indices]

    # Sample pupil data with target rows
    pupil_indices = np.linspace(0, len(pupil_df) - 1, num=target_rows, dtype=int)
    sampled_pupil = pupil_df.iloc[pupil_indices]

    # Combine gaze and pupil data
    combined_data = pd.concat([sampled_gaze.reset_index(drop=True), sampled_pupil.reset_index(drop=True)], axis=1)

    num_per_window = int(window_time * fps)
    overlaps = int(num_per_window * overlap)
    stride = num_per_window - overlaps

    batches = []

    end_index = len(combined_data)

    while end_index > 0:
        start_index = max(0, end_index - num_per_window)
        batch = combined_data.iloc[start_index:end_index]

        # Only append if batch size is consistent
        if len(batch) == num_per_window:
            batches.append(batch)

        end_index -= stride

        if start_index == 0:
            break
    
    combined_array = np.array([batch.values for batch in batches])

    return combined_array