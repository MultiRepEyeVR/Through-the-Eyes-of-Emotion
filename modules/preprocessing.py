from numpy.fft import fft, ifft, fftfreq
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# Low-Pass Filter
def apply_LPF(sampled_data, sampling_frequency=120, cutoff_frequency=5):
    sampling_period = 1 / sampling_frequency  # in seconds

    # Fourier transform and low pass filtering
    fft_data = fft(sampled_data)
    low_pass_mask = abs(fftfreq(len(sampled_data), d=sampling_period)) <= cutoff_frequency
    filtered_fft_data = fft_data * low_pass_mask
    filtered_data2 = ifft(filtered_fft_data).real

    return filtered_data2


# Process Pupil Data
def filter_pupil_data(pupil_data, target_length):
    pupil_data = pupil_data.copy()
    original_length = len(pupil_data)

    # Handle low confidence points first
    confidence_mask = pupil_data['confidence'] > 0.6
    all_indices = np.arange(original_length)
    valid_indices = np.where(confidence_mask)[0]
    pupil_data['diameter'] = np.interp(all_indices, valid_indices, pupil_data.loc[confidence_mask, 'diameter'])

    # Compute the MAD and filter
    pupil_data['derivative'] = pupil_data['diameter'].diff().fillna(0)
    median = np.median(pupil_data['derivative'])
    mad = np.median(np.abs(pupil_data['derivative'] - median))
    mad_threshold = 3 * mad
    mad_mask = ~(np.abs(pupil_data['derivative'] - median) > mad_threshold)
    valid_indices = np.where(mad_mask)[0]
    pupil_data['diameter'] = np.interp(all_indices, valid_indices, pupil_data.loc[mad_mask, 'diameter'])

    # Z-score filtering
    window_size = 120
    threshold = 0.1
    mean = pupil_data['diameter'].rolling(window=window_size, center=True).mean()
    std = pupil_data['diameter'].rolling(window=window_size, center=True).std()
    z_score_mask = (pupil_data['diameter'] - mean).abs() <= threshold * std
    valid_indices = np.where(z_score_mask)[0]
    pupil_data['diameter'] = np.interp(all_indices, valid_indices, pupil_data.loc[z_score_mask, 'diameter'])

    # Apply low-pass filter
    filtered_diameter = apply_LPF(pupil_data['diameter'], sampling_frequency=120, cutoff_frequency=5)

    # Resample data
    original_time = np.linspace(0, original_length / 120, num=original_length, endpoint=False)
    new_time = np.linspace(0, original_length / 120, num=target_length, endpoint=False)
    resampled_diameter = np.interp(new_time, original_time, filtered_diameter)

    return resampled_diameter


# Process Gaze Data
def filter_gaze_data(gaze_data):
    gaze_data = gaze_data.copy()
    original_length = len(gaze_data)

    # Create a mask for data above confidence threshold
    confidence_mask = gaze_data['confidence'] > 0.6
    all_indices = np.arange(original_length)
    high_confidence_indices = np.where(confidence_mask)[0]

    # First pass of interpolation for low-confidence points
    for coord in ['norm_pos_x', 'norm_pos_y']:
        gaze_data[coord] = np.interp(all_indices, high_confidence_indices, gaze_data.loc[confidence_mask, coord])

    # Process each coordinate independently for MAD
    for coord in ['norm_pos_x', 'norm_pos_y']:
        # Compute the MAD
        gaze_data[f'derivative_{coord}'] = gaze_data[coord].diff().fillna(0)
        median = np.median(gaze_data[f'derivative_{coord}'])
        mad = np.median(np.abs(gaze_data[f'derivative_{coord}'] - median))
        mad_threshold = 3 * mad  # Define outlier threshold

        # Create a mask for non-outliers based on MAD
        mad_mask = ~((gaze_data[f'derivative_{coord}'] - median).abs() > mad_threshold)
        valid_mad_indices = np.where(mad_mask)[0]

        # Second pass of interpolation for MAD outliers
        gaze_data[coord] = np.interp(all_indices, valid_mad_indices, gaze_data.loc[mad_mask, coord])

    return gaze_data[['norm_pos_x', 'norm_pos_y']].reset_index(drop=True)


def preprocess_gaze_pupil_data(gaze_path, pupil_path, save_path=None):
    # ==========================Gaze Data==========================
    gaze_data = pd.read_csv(gaze_path)
    gaze_unfiltered = gaze_data[['confidence', 'norm_pos_x', 'norm_pos_y']]

    filtered_gaze = filter_gaze_data(gaze_unfiltered)

    gaze_standardizer = StandardScaler()
    gaze_normalizer = MinMaxScaler()
    filtered_gaze[['norm_pos_x', 'norm_pos_y']] = gaze_standardizer.fit_transform(filtered_gaze[['norm_pos_x', 'norm_pos_y']])
    filtered_gaze[['norm_pos_x', 'norm_pos_y']] = gaze_normalizer.fit_transform(filtered_gaze[['norm_pos_x', 'norm_pos_y']])
    # ==========================Pupil Data==========================
    pupil_data = pd.read_csv(pupil_path)
    eye1_unfiltered_pupil = pupil_data[pupil_data['eye_id'] == 1][['confidence', 'diameter']].reset_index(drop=True)
    eye0_unfiltered_pupil = pupil_data[pupil_data['eye_id'] == 0][['confidence', 'diameter']].reset_index(drop=True)

    # Gaze original sampling rate 240 Hz, and pupil original sampling rate 120 Hz
    target_length_eye1 = 2 * len(eye1_unfiltered_pupil) - 1
    target_length_eye0 = 2 * len(eye0_unfiltered_pupil) - 1

    filtered_eye1_pupil = filter_pupil_data(eye1_unfiltered_pupil, target_length_eye1)
    filtered_eye0_pupil = filter_pupil_data(eye0_unfiltered_pupil, target_length_eye0)

    min_pupil_length = min(len(filtered_eye1_pupil), len(filtered_eye0_pupil))

    filtered_eye1_diameter = filtered_eye1_pupil[:min_pupil_length]
    filtered_eye0_diameter = filtered_eye0_pupil[:min_pupil_length]

    filtered_pupil = pd.DataFrame({
        'eye0_diameter': filtered_eye0_diameter,
        'eye1_diameter': filtered_eye1_diameter
    })

    eye_standardizer = StandardScaler()
    eye_normalizer = MinMaxScaler()
    filtered_pupil[['eye0_diameter', 'eye1_diameter']] = eye_standardizer.fit_transform(filtered_pupil[['eye0_diameter', 'eye1_diameter']])
    filtered_pupil[['eye0_diameter', 'eye1_diameter']] = eye_normalizer.fit_transform(filtered_pupil[['eye0_diameter', 'eye1_diameter']])

    # ===========================Save==========================
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        filtered_gaze.to_csv(save_path + '/filtered_gaze.csv', index=False, mode='w')
        filtered_pupil.to_csv(save_path + '/filtered_eye.csv', index=False, mode='w')

    return filtered_gaze, filtered_pupil
