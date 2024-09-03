import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import biosppy
from scipy.fftpack import fft
import pandas as pd
import numpy as np
import biosppy
from biosppy.signals import ppg
from biosppy.signals.tools import filter_signal
from scipy.signal import find_peaks, butter, filtfilt
import librosa
import neurokit2 as nk

SAMPLING_RATE = 25
LOWCUT = 0.2
HIGHCUT = 5.0
SYNT_HEART_RATE = 70
SYNT_DURATION = 20
N_FFT=512
NOISE_LEVEL = 0.3

# CLASSIFICATION WINDOW AND STEP
SEGMENT_LENGTH_S = 20  # 20 seconds per segment
STEP_SIZE_S = 5        # 5 seconds step size

def generate_synthetic_ppg(duration, sampling_rate, heart_rate):
    ppg_signal = nk.ppg_simulate(duration, sampling_rate, heart_rate, random_state=42)
    return ppg_signal

def add_noise(signal, noise_level):
    noise = np.random.normal(0, noise_level, signal.shape)
    return signal + noise

def process_ppg_with_biosppy(signal, sampling_rate):
    ppg_obj = biosppy.signals.ppg.ppg(signal, sampling_rate=sampling_rate, show=False)
    heart_rate = np.mean(ppg_obj['heart_rate'])
    return ppg_obj, heart_rate

def process_ppg(signal, sampling_rate):
    signals, info = nk.ppg_process(signal, sampling_rate=sampling_rate)
    return signals, info

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def insert_discontinuities(df, time_col, threshold):
    df = df.copy()
    # Calculate the time differences
    time_diff = df[time_col].diff().abs()
    # Identify where the time difference exceeds the threshold
    discontinuity_indices = time_diff[time_diff > threshold].index

    # Insert None values
    for index in discontinuity_indices:
        df = pd.concat([df.iloc[:index], pd.DataFrame({time_col: [None], 'ledGreen_filtered': [None], 'ledGreen': [None]}), df.iloc[index:]]).reset_index(drop=True)
    
    return df

def calculate_snr(signal, sampling_rate, signal_band, noise_band, window='hann', n_fft=None):
    N = len(signal)
    if n_fft is None:
        n_fft = N

    # Apply window function
    if window == 'hann':
        window_func = np.hanning(N)
    elif window == 'hamming':
        window_func = np.hamming(N)
    elif window == 'blackman':
        window_func = np.blackman(N)
    else:
        window_func = np.ones(N)
    
    # Normalize the windowed signal to maintain power
    windowed_signal = signal * window_func
    windowed_signal = windowed_signal / np.sqrt(np.mean(window_func**2))

    # FFT and Power Spectral Density (PSD)
    fft_result = np.fft.fft(windowed_signal, n=n_fft)
    fft_result = fft_result[:n_fft // 2]  # Consider only positive frequencies
    psd = (np.abs(fft_result)**2) / (N * np.sum(window_func**2))

    # Calculate power in the signal and noise bands
    signal_power = np.sum(psd[signal_band])
    noise_power = np.sum(psd[noise_band])

    snr = signal_power / noise_power
    snr_db = 10 * np.log10(snr)
    return snr_db

def determine_snr_threshold(synthetic_signal, sampling_rate, noise_level, window='hann', n_fft=None):
    noisy_signal = add_noise(synthetic_signal, noise_level)
    if n_fft is None:
        n_fft = len(noisy_signal)
    
    frequencies = np.fft.fftfreq(n_fft, 1/sampling_rate)[:n_fft // 2]
    # Define signal and noise bands
    signal_band = (frequencies >= LOWCUT) & (frequencies <= HIGHCUT)  # Typical PPG signal range
    noise_band = (frequencies < LOWCUT) | (frequencies > HIGHCUT)     # Example noise band
    
    snr_db = calculate_snr(noisy_signal, sampling_rate, signal_band, noise_band, window=window, n_fft=n_fft)
    return snr_db

# Segment the actual PPG signal
def segment_signal(signal, segment_length):
    return [signal[i:i + segment_length] for i in range(0, len(signal), segment_length)]

# Evaluate actual PPG signal segments
def evaluate_ppg_signal(actual_signal, sampling_rate, snr_threshold, segment_length):
    segments = segment_signal(actual_signal, segment_length)
    segment_results = []
    for segment in segments:
        if len(segment) < segment_length:
            continue  # Skip segments that are too short
        snr = calculate_snr(segment, sampling_rate)
        quality = 'good' if snr >= snr_threshold else 'bad'
        segment_results.append((snr, quality))
    return segment_results



def get_good_bad_segments(df,snr_threshold,sampling_rate,n_fft=512, window_duration=20):
    actual_ppg = df['ledGreen_filtered'].values

    # Segment the actual PPG signals
    segment_length = window_duration * sampling_rate  # Segment length in samples (10 seconds)
    segments = [actual_ppg[i:i + segment_length] for i in range(0, len(actual_ppg), segment_length)]

    # Define signal and noise frequency bands
    frequencies = np.fft.fftfreq(n_fft, 1/sampling_rate)[:n_fft // 2]
    signal_band = (frequencies >= LOWCUT) & (frequencies <= HIGHCUT)  # PPG signal band
    noise_band = (frequencies < LOWCUT) | (frequencies > HIGHCUT)     # Noise band

    # Evaluate each segment and classify them based on SNR
    good_bad_segments = []
    for i, segment in enumerate(segments):
        if len(segment) < segment_length:
            continue  # Skip segments that are too short
        #print(len(segment))
        #segment_results = biosppy.signals.ppg.ppg(signal=segment, sampling_rate=25, show=False)
        #segment = segment_results['filtered']
        snr = calculate_snr(segment, sampling_rate, signal_band, noise_band,'hann',n_fft)
        #print(snr)
        quality = 'good' if snr >= snr_threshold else 'bad'
        #print(quality)
        start_idx = i * segment_length
        end_idx = start_idx + len(segment)
        good_bad_segments.append((quality, start_idx, end_idx))

# Function to segment signal into overlapping windows
def overlapping_windows(signal, segment_length, step_size):
    return [signal[i:i + segment_length] for i in range(0, len(signal) - segment_length + 1, step_size)]

# Function to aggregate scores from overlapping segments
def aggregate_classifications(classifications, signal_length, segment_length, step_size):
    classification = np.zeros(signal_length)
    counts = np.zeros(signal_length)

    for quality, start_idx, end_idx in classifications:
        value = 1 if quality == 'good' else -1
        classification[start_idx:end_idx] += value
        counts[start_idx:end_idx] += 1

    # Aggregate scores without normalizing
    return classification

def remove_segments_based_on_score(df_original, aggregated_score, threshold):
    # Identify the indices of bad segments based on the aggregate score
    bad_indices = np.where(aggregated_score < threshold)[0]
    
    # Create ranges of continuous bad segments
    bad_ranges = []
    if len(bad_indices) > 0:
        start_idx = bad_indices[0]
        for i in range(1, len(bad_indices)):
            if bad_indices[i] != bad_indices[i-1] + 1:
                bad_ranges.append((start_idx, bad_indices[i-1]))
                start_idx = bad_indices[i]
        bad_ranges.append((start_idx, bad_indices[-1]))

    # Remove the bad segments from the original DataFrame
    for start_idx, end_idx in bad_ranges:
        df_original = df_original.drop(index=range(start_idx, end_idx+1))

    return df_original.reset_index(drop=True)

def align_sensors_on_time(df, sensor_columns):
    """Aligns all sensor data by interpolating to common timestamps."""
    # Ensure 'unixTimes' is numeric
    df['unixTimes'] = pd.to_numeric(df['unixTimes'], errors='coerce')
    
    # Set Unix times as index
    df.set_index('unixTimes', inplace=True)
    
    # Interpolate missing values to align all sensor data
    #df[sensor_columns] = df[sensor_columns].interpolate(method='linear', axis=0)
    
    # Drop any remaining rows with NaN values after interpolation
    #df.dropna(subset=sensor_columns, inplace=True)
    
    return df.reset_index()

def filter_good_bad_segments(df_original, config,seed=42):
    np.random.seed(seed)
    
    SAMPLING_RATE = config['frequencies']['ppg']
    SEGMENT_LENGTH_S = config['segmentation']['segment_length_seconds']
    STEP_SIZE_S = config['segmentation']['step_size_seconds']

    LOWCUT = config['filters']['ppg']['lowcut']
    HIGHCUT = config['filters']['ppg']['highcut']
    
    SYNT_HEART_RATE = config['synthetic_data']['heart_rate']
    SYNT_DURATION = config['synthetic_data']['duration']
    N_FFT = config['synthetic_data']['n_fft']
    NOISE_LEVEL = config['synthetic_data']['noise_level']

    df = df_original.copy()
    df = df.dropna(subset=['ledIR', 'ledRed', 'ledGreen', 'sleep_label'])
    df['ledGreen_filtered']=bandpass_filter(df['ledGreen'].dropna(), LOWCUT, HIGHCUT, SAMPLING_RATE)

    
    # SYNTHETIC PPG
    synthetic_ppg = generate_synthetic_ppg(SYNT_DURATION, SAMPLING_RATE, SYNT_HEART_RATE)
    synthetic_ppg=bandpass_filter(synthetic_ppg, LOWCUT, HIGHCUT, SAMPLING_RATE)
    #SNR THRESHOLD
    snr_threshold = determine_snr_threshold(synthetic_ppg, SAMPLING_RATE, NOISE_LEVEL,'hann',N_FFT)
    print("snr_threshold: ",snr_threshold)
    
    # CLASSIFICATION SEGMENT
    segment_length = SEGMENT_LENGTH_S * SAMPLING_RATE  # Segment length in samples
    step_size = STEP_SIZE_S * SAMPLING_RATE       
    segments = overlapping_windows(df['ledGreen_filtered'].values, segment_length, step_size)
    print("Overlapping segments created")
    print(len(segments))
    
    good_bad_segments_overlap = []
    frequencies = np.fft.fftfreq(N_FFT, 1/SAMPLING_RATE)[:N_FFT // 2]
    signal_band = (frequencies >= LOWCUT) & (frequencies <= HIGHCUT)  # PPG signal band
    noise_band = (frequencies < LOWCUT) | (frequencies > HIGHCUT)     # Noise band
    for i, segment in enumerate(segments):
        if len(segment) < segment_length:
            continue  # Skip segments that are too short
        snr = calculate_snr(segment, SAMPLING_RATE, signal_band, noise_band, 'hann', N_FFT)
        quality = 'good' if snr >= snr_threshold else 'bad'
        start_idx = i * step_size
        end_idx = start_idx + segment_length
        good_bad_segments_overlap.append((quality, start_idx, end_idx))
    print("segments Classified into good or bad")
    
     # Aggregate the classifications for each signal value
    aggregated_score = aggregate_classifications(good_bad_segments_overlap, 
                                                 len(df['ledGreen_filtered']), 
                                                 segment_length, step_size)
    
    print("aggregated_score created")
    # Define a threshold for bad segments
    threshold = 0  

    # Remove bad segments from the original DataFrame based on the aggregate score
    df_filtered_original = remove_segments_based_on_score(df_original, aggregated_score, threshold)
    print("Filter df based on aggregated_score")
    
    sensor_columns_ppg = config['features']['ppg']
    sensor_columns_gyro = config['features']['gyroscope']
    sensor_columns_acc = config['features']['accelerometer']
    
    # Align all sensors by Unix times
    df_aligned = align_sensors_on_time(df_filtered_original, sensor_columns_ppg + sensor_columns_gyro + sensor_columns_acc)
    print("df_aligned ready")
    
    return df_aligned

if __name__ == '__main__':
    np.random.seed(42)
    
    #READ ACTUAL PPG
    df_original = pd.read_csv('data/train/california-00011978-right-sync.csv')
    #df_original = pd.read_csv('data/adhd_train/adhd-KKI_004-left-sync.csv')
    df_original['sleep_label'] = df_original['sleep_stage'].apply(lambda x: 0 if x == 'WK' else 1)  # 0: awake, 1: sleep
    df = df_original.copy()
    
    df = df.dropna(subset=['ledIR', 'ledRed', 'ledGreen', 'sleep_label'])
    df['ledGreen_filtered']=bandpass_filter(df['ledGreen'].dropna(), LOWCUT, HIGHCUT, SAMPLING_RATE)

    # SYNTHETIC PPG
    synthetic_ppg = generate_synthetic_ppg(SYNT_DURATION, SAMPLING_RATE, SYNT_HEART_RATE)
    synthetic_ppg=bandpass_filter(synthetic_ppg, LOWCUT, HIGHCUT, SAMPLING_RATE)
    
    #SNR THRESHOLD
    snr_threshold = determine_snr_threshold(synthetic_ppg, SAMPLING_RATE, NOISE_LEVEL,'hann',N_FFT)
    print("snr_threshold: ",snr_threshold)
    

    # CLASSIFICATION SEGMENT
    segment_length = SEGMENT_LENGTH_S * SAMPLING_RATE  # Segment length in samples
    step_size = STEP_SIZE_S * SAMPLING_RATE       
    segments = overlapping_windows(df['ledGreen_filtered'].values, segment_length, step_size)
    
    good_bad_segments_overlap = []
    frequencies = np.fft.fftfreq(N_FFT, 1/SAMPLING_RATE)[:N_FFT // 2]
    signal_band = (frequencies >= LOWCUT) & (frequencies <= HIGHCUT)  # PPG signal band
    noise_band = (frequencies < LOWCUT) | (frequencies > HIGHCUT)     # Noise band
    for i, segment in enumerate(segments):
        if len(segment) < segment_length:
            continue  # Skip segments that are too short
        snr = calculate_snr(segment, SAMPLING_RATE, signal_band, noise_band, 'hann', N_FFT)
        quality = 'good' if snr >= snr_threshold else 'bad'
        start_idx = i * step_size
        end_idx = start_idx + segment_length
        good_bad_segments_overlap.append((quality, start_idx, end_idx))

   
    # Aggregate the classifications for each signal value
    aggregated_score = aggregate_classifications(good_bad_segments_overlap, 
                                                 len(df['ledGreen_filtered']), 
                                                 segment_length, step_size)


    # Define a threshold for bad segments
    threshold = 0  

    # Remove bad segments from the original DataFrame based on the aggregate score
    df_filtered_original = remove_segments_based_on_score(df_original, aggregated_score, threshold)

    print("Original DataFrame shape:", df_original.shape)
    print("Filtered DataFrame shape:", df_filtered_original.shape)
    
    print(df_original.dropna(subset=['ledIR', 'ledRed', 'ledGreen', 'sleep_label']).shape)
    print(df_original.dropna(subset=['gyroscopeX', 'gyroscopeY', 'gyroscopeZ', 'sleep_label']).shape)
    print(df_original.dropna(subset=['accelerometerX', 'gyroscopeY', 'accelerometerY', 'sleep_label']).shape)
    print(df_original.dropna(subset=['tempObject']).shape)
    
    print(df_filtered_original.dropna(subset=['ledIR', 'ledRed', 'ledGreen', 'sleep_label']).shape)
    print(df_filtered_original.dropna(subset=['gyroscopeX', 'gyroscopeY', 'gyroscopeZ', 'sleep_label']).shape)
    print(df_filtered_original.dropna(subset=['accelerometerX', 'gyroscopeY', 'accelerometerY', 'sleep_label']).shape)
    print(df_filtered_original.dropna(subset=['tempObject']).shape)
    
    sensor_columns_ppg = ['ledIR', 'ledRed', 'ledGreen']
    sensor_columns_gyro = ['gyroscopeX', 'gyroscopeY', 'gyroscopeZ']
    sensor_columns_acc = ['accelerometerX', 'accelerometerY', 'accelerometerZ']

    # Align all sensors by Unix times
#     df_aligned = align_sensors_on_time(df_filtered_original, sensor_columns_ppg + sensor_columns_gyro + sensor_columns_acc)

#     print(df_aligned.dropna(subset=['ledIR', 'ledRed', 'ledGreen', 'sleep_label']).shape)
#     print(df_aligned.dropna(subset=['gyroscopeX', 'gyroscopeY', 'gyroscopeZ', 'sleep_label']).shape)
#     print(df_aligned.dropna(subset=['accelerometerX', 'gyroscopeY', 'accelerometerY', 'sleep_label']).shape)
    
#     good_mask = aggregated_score >= 0
#     df_filtered = df[good_mask].reset_index(drop=True)
    
#     print("Original DataFrame shape:", df.shape)
#     print("Filtered DataFrame shape:", df_filtered.shape)
    
    
    
    
    
    