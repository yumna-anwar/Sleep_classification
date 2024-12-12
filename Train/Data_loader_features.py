import pandas as pd
import numpy as np
import os
import tensorflow as tf
import json
from collections import Counter 
from sklearn.utils import class_weight
from scipy.signal import butter, filtfilt
from ppg_preprocess import filter_good_bad_segments
from sklearn.preprocessing import RobustScaler,MinMaxScaler,StandardScaler
from scipy.interpolate import interp1d
import gc
import neurokit2 as nk
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
#from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


MASTER_FEATURE_KEYS = [
    "HRV_MeanNN", "HRV_SDNN", "HRV_SDANN1", "HRV_SDNNI1", "HRV_SDANN2", "HRV_SDNNI2",
    "HRV_SDANN5", "HRV_SDNNI5", "HRV_RMSSD", "HRV_SDSD", "HRV_CVNN", "HRV_CVSD",
    "HRV_MedianNN", "HRV_MadNN", "HRV_MCVNN", "HRV_IQRNN", "HRV_SDRMSSD",
    "HRV_Prc20NN", "HRV_Prc80NN", "HRV_pNN50", "HRV_pNN20", "HRV_MinNN",
    "HRV_MaxNN", "HRV_HTI", "HRV_TINN", "HRV_ULF", "HRV_VLF", "HRV_LF", "HRV_HF",
    "HRV_VHF", "HRV_TP", "HRV_LFHF", "HRV_LFn", "HRV_HFn", "HRV_LnHF",
    "HRV_SD1", "HRV_SD2", "HRV_SD1SD2", "HRV_S", "HRV_CSI", "HRV_CVI",
    "HRV_CSI_Modified", "HRV_PIP", "HRV_IALS", "HRV_PSS", "HRV_PAS", "HRV_GI",
    "HRV_SI", "HRV_AI", "HRV_PI", "HRV_C1d", "HRV_C1a", "HRV_SD1d", "HRV_SD1a",
    "HRV_C2d", "HRV_C2a", "HRV_SD2d", "HRV_SD2a", "HRV_Cd", "HRV_Ca",
    "HRV_SDNNd", "HRV_SDNNa", "HRV_DFA_alpha1", "HRV_MFDFA_alpha1_Width",
    "HRV_MFDFA_alpha1_Peak", "HRV_MFDFA_alpha1_Mean", "HRV_MFDFA_alpha1_Max",
    "HRV_MFDFA_alpha1_Delta", "HRV_MFDFA_alpha1_Asymmetry",
    "HRV_MFDFA_alpha1_Fluctuation", "HRV_MFDFA_alpha1_Increment", "HRV_ApEn",
    "HRV_SampEn", "HRV_ShanEn", "HRV_FuzzyEn", "HRV_MSEn", "HRV_CMSEn",
    "HRV_RCMSEn", "HRV_CD", "HRV_HFD", "HRV_KFD", "HRV_LZC", "Min HR", "Max HR",
    "Mean HR", "Mean PPI", "SEM PPI", "pPP50", "RIAM", "RIFM", "VLF Power",
    "LF Power", "HF Power", "LF/HF Ratio"
]
# FILTERS
# High-pass and band-pass filters
def butter_bandpass(lowcut, highcut, fs, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)

def butter_highpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    high = cutoff / nyquist
    b, a = butter(order, high, btype='highpass')
    return b, a

def highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    return filtfilt(b, a, data)

def upsample_signal(data, original_fs, target_fs):
    time_original = np.arange(len(data)) / original_fs
    num_samples = int(len(data) * (target_fs / original_fs))
    time_upsampled = np.linspace(0, time_original[-1], num=num_samples)
    interpolator = interp1d(time_original, data, kind='linear')
    return interpolator(time_upsampled)

# PROCESS EACH SENSOR DATA

def remove_peaks_and_interpolate(sensor_df, feature, lower_percentile=0.01, upper_percentile=0.99):
    # Calculate the lower and upper bounds for the feature based on the percentiles
    lower_bound = sensor_df[feature].quantile(lower_percentile)
    upper_bound = sensor_df[feature].quantile(upper_percentile)
    
    # Identify the peaks (values outside the quartile bounds)
    peak_mask = (sensor_df[feature] < lower_bound) | (sensor_df[feature] > upper_bound)
    
    # Create a copy of the data
    clean_data = sensor_df[feature].copy()
    
    # Get indices for interpolation
    peak_indices = clean_data[peak_mask].index
    valid_indices = clean_data[~peak_mask].index
    
    # Perform linear interpolation to fill peak values
    interpolator = interp1d(valid_indices, clean_data[~peak_mask], bounds_error=False, fill_value="extrapolate")
    clean_data[peak_indices] = interpolator(peak_indices)
    
    return clean_data

def preprocess_sensor_data(df, features, lowcut=None, highcut=None, fs=25,verbos=False):
    sensor_df = df[['unixTimes'] + features + ['sleep_label']].dropna()
    sensor_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    sensor_df.dropna(inplace=True)
    
    # Apply bandpass filter to each feature
    if lowcut!=None and highcut!=None:
        for feature in features:
            sensor_df[feature] = remove_peaks_and_interpolate(sensor_df, feature)
            sensor_df[feature] = bandpass_filter(sensor_df[feature], lowcut, highcut, fs)
    elif lowcut!=None and highcut==None:
        for feature in features:
            sensor_df[feature] = remove_peaks_and_interpolate(sensor_df, feature)
            sensor_df[feature] = highpass_filter(sensor_df[feature], lowcut, fs,order=5)
    
    #scaler = StandardScaler()
    scaler = RobustScaler()
    for feature in features:
        sensor_df[feature] = scaler.fit_transform(sensor_df[feature].values.reshape(-1, 1))
    return sensor_df

def preprocess_sensor_data_ppg(df, features, lowcut, highcut, original_fs, target_fs):
    sensor_df = df[features].dropna()
    sensor_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    sensor_df.dropna(inplace=True)

    upsampled_data = {}
    for feature in features:
        upsampled = upsample_signal(sensor_df[feature].values, original_fs, target_fs)
        filtered = bandpass_filter(upsampled, lowcut, highcut, target_fs)
        scaler = RobustScaler()
        scaled = scaler.fit_transform(filtered.reshape(-1, 1)).flatten()
        upsampled_data[feature] = scaled
    # Create a new DataFrame for upsampled data
    upsampled_df = pd.DataFrame(upsampled_data)
    return upsampled_df

def _extract_ppg_features(ppg_signal, target_fs):
    ppg_cleaned = nk.ppg_clean(ppg_signal, sampling_rate=target_fs)
    peaks_info = nk.ppg_findpeaks(ppg_cleaned, sampling_rate=target_fs)
    ppg_peaks = peaks_info["PPG_Peaks"]
    if not peaks_info["PPG_Peaks"].size:
            return {f"PPG_{metric}": np.nan for metric in ["Min HR", "Max HR", "Mean HR", "HRV_SDNN"]}

    heart_rate = nk.ppg_rate(peaks_info, sampling_rate=target_fs, desired_length=len(ppg_cleaned))

    min_hr = np.min(heart_rate)
    max_hr = np.max(heart_rate)
    mean_hr = np.mean(heart_rate)
    hrv_metrics = nk.hrv(peaks=peaks_info["PPG_Peaks"], sampling_rate=target_fs, show=False)

    # Step 6: Convert HRV metrics to a dictionary
    hrv_features = hrv_metrics.to_dict('records')[0] if not hrv_metrics.empty else {}

    # Step 7: Add min, max, and mean heart rate to the features
    hrv_features.update({
        "Min HR": min_hr,
        "Max HR": max_hr,
        "Mean HR": mean_hr
    })
    
    
    # PPI (PPG peak-to-peak interval)
    ppi_intervals = np.diff(ppg_peaks) / target_fs  # Convert peak intervals to seconds
    mean_ppi = np.mean(ppi_intervals)
    sem_ppi = np.std(ppi_intervals, ddof=1) / np.sqrt(len(ppi_intervals))  # Standard error of the mean PPI

    # pPP50: Percentage of PPI differences greater than 50 ms
    ppi_diffs = np.abs(np.diff(ppi_intervals))
    pPP50 = np.sum(ppi_diffs > 0.05) / len(ppi_diffs) * 100

    # Respiratory modulation features (RIAM and RIFM) - Placeholder methods
    # Respiratory-Induced Amplitude Modulation (RIAM)
    amplitude = ppg_cleaned[ppg_peaks]
    riam = np.std(amplitude) / np.mean(amplitude) if np.mean(amplitude) > 0 else np.nan

    # Respiratory-Induced Frequency Modulation (RIFM)
    rifm = np.std(ppi_intervals) / mean_ppi if mean_ppi > 0 else np.nan

    # Frequency-based features (VLF, LF, HF, and LF/HF ratio)
    freqs, power = welch(ppg_cleaned, fs=target_fs, nperseg=1024)
    vlf_band = (0.003, 0.04)
    lf_band = (0.04, 0.15)
    hf_band = (0.15, 0.4)

    vlf_power = np.trapz(power[(freqs >= vlf_band[0]) & (freqs < vlf_band[1])])
    lf_power = np.trapz(power[(freqs >= lf_band[0]) & (freqs < lf_band[1])])
    hf_power = np.trapz(power[(freqs >= hf_band[0]) & (freqs < hf_band[1])])

    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else np.nan

    # Combine features into the feature dictionary
    hrv_features.update({
        "Mean PPI": mean_ppi,
        "SEM PPI": sem_ppi,
        "pPP50": pPP50,
        "RIAM": riam,
        "RIFM": rifm,
        "VLF Power": vlf_power,
        "LF Power": lf_power,
        "HF Power": hf_power,
        "LF/HF Ratio": lf_hf_ratio
    })
    
    return hrv_features

def extract_ppg_features(ppg_signal, target_fs, channel_name, verbos=False):
    try:
        # Step 1: Clean PPG signal and detect peaks
        ppg_cleaned = nk.ppg_clean(ppg_signal, sampling_rate=target_fs)
        peaks_info = nk.ppg_findpeaks(ppg_cleaned, sampling_rate=target_fs)
        ppg_peaks = peaks_info.get("PPG_Peaks", [])
        if verbos:
            print("num of peaks",len(ppg_peaks))
        # Step 2: Handle cases where no peaks or insufficient peaks are detected
        if len(ppg_peaks) < 30:  # Arbitrary threshold, can be adjusted based on requirements
            raise ValueError(f"Insufficient peaks detected in channel {channel_name} (found {len(ppg_peaks)})")

        # Step 3: Calculate heart rate metrics
        heart_rate = nk.ppg_rate(peaks_info, sampling_rate=target_fs, desired_length=len(ppg_cleaned))
        min_hr = np.min(heart_rate)
        max_hr = np.max(heart_rate)
        mean_hr = np.mean(heart_rate)

        # Step 4: Calculate HRV metrics
        hrv_metrics = nk.hrv(peaks=ppg_peaks, sampling_rate=target_fs, show=False)
        hrv_features = hrv_metrics.to_dict('records')[0] if not hrv_metrics.empty else {}

        # Step 5: Add heart rate metrics
        hrv_features.update({
            "Min HR": min_hr,
            "Max HR": max_hr,
            "Mean HR": mean_hr
        })

        # Step 6: Calculate PPI metrics
        ppi_intervals = np.diff(ppg_peaks) / target_fs  # Convert peak intervals to seconds
        mean_ppi = np.mean(ppi_intervals)
        sem_ppi = np.std(ppi_intervals, ddof=1) / np.sqrt(len(ppi_intervals))
        ppi_diffs = np.abs(np.diff(ppi_intervals))
        pPP50 = np.sum(ppi_diffs > 0.05) / len(ppi_diffs) * 100

        # Step 7: Calculate respiratory modulation features
        amplitude = ppg_cleaned[ppg_peaks]
        riam = np.std(amplitude) / np.mean(amplitude) if np.mean(amplitude) > 0 else np.nan
        rifm = np.std(ppi_intervals) / mean_ppi if mean_ppi > 0 else np.nan

        # Step 8: Calculate frequency-based metrics
        freqs, power = welch(ppg_cleaned, fs=target_fs, nperseg=1024)
        vlf_band = (0.003, 0.04)
        lf_band = (0.04, 0.15)
        hf_band = (0.15, 0.4)

        vlf_power = np.trapz(power[(freqs >= vlf_band[0]) & (freqs < vlf_band[1])])
        lf_power = np.trapz(power[(freqs >= lf_band[0]) & (freqs < lf_band[1])])
        hf_power = np.trapz(power[(freqs >= hf_band[0]) & (freqs < hf_band[1])])
        lf_hf_ratio = lf_power / hf_power if hf_power > 0 else np.nan

        # Combine all features into the dictionary
        hrv_features.update({
            "Mean PPI": mean_ppi,
            "SEM PPI": sem_ppi,
            "pPP50": pPP50,
            "RIAM": riam,
            "RIFM": rifm,
            "VLF Power": vlf_power,
            "LF Power": lf_power,
            "HF Power": hf_power,
            "LF/HF Ratio": lf_hf_ratio
        })

    except Exception as e:
        print(f"Error in {channel_name} feature extraction: {e}")
        # Use a predefined list of all metrics
        hrv_features = {f"{channel_name}_{metric}": np.nan for metric in MASTER_FEATURE_KEYS}

    # Ensure all keys from MASTER_FEATURE_KEYS are present
    full_features = {f"{channel_name}_{metric}": hrv_features.get(metric, np.nan) for metric in MASTER_FEATURE_KEYS}
    return full_features


def extract_statistical_features(windowed_data, feature_names):
    results = {}
    
    for i, feature in enumerate(feature_names):
        # Extract the data for the current feature
        feature_data = windowed_data[:, i].flatten()  # Assuming windowed_data is a NumPy array

        # Calculate statistics for each feature
        stats = {
            f"{feature}_mean": np.mean(feature_data),
            f"{feature}_median": np.median(feature_data),
            f"{feature}_std": np.std(feature_data),
            f"{feature}_var": np.var(feature_data),
            f"{feature}_min": np.min(feature_data),
            f"{feature}_max": np.max(feature_data),
            f"{feature}_range": np.max(feature_data) - np.min(feature_data),
            f"{feature}_skew": skew(feature_data),
            f"{feature}_kurtosis": kurtosis(feature_data)
        }
        
        # Update results with this feature's statistics
        results.update(stats)

    # Convert results to DataFrame (or dictionary depending on usage)
    return results

# Load configuration
def load_config(config_file='config_features.json'):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config


# Calculate the frequency of data collection
def calculate_frequency(data):
    time_diffs = np.diff(data['unixTimes'].values)  # Calculate time differences between successive samples
    avg_time_diff = np.mean(time_diffs)  # Average time difference
    frequency = 1000 / avg_time_diff  # Frequency is the inverse of the average time difference
    return frequency




def create_combined_tf_dataset(generators, output_types, output_shapes):
    def combined_gen():
        sensor_generators = [gen() for gen in generators]
        while True:
            try:
                features = [next(sensor_gen) for sensor_gen in sensor_generators]
                yield tuple(f[0] for f in features), features[0][1]
            except StopIteration:
                break
    
    return tf.data.Dataset.from_generator(
        combined_gen,
        output_types=output_types,
        output_shapes=output_shapes
    )


def resample_data(df, target_freq, columns_to_resample, time_col='unixTimes', max_gap=1*60000, verbos=False):  # max_gap in milliseconds
    df = df.copy()
    df['datetime'] = pd.to_datetime(df[time_col], unit='ms')
    
    # Identify large gaps
    df['time_diff'] = df['datetime'].diff().dt.total_seconds() * 1000  # in milliseconds
    large_gaps = df['time_diff'] > max_gap
    
    if verbos:
        print("Number of large gaps identified:", sum(large_gaps))
    
    # Initialize an empty list to collect resampled segments
    resampled_segments = []
    
    # Start index for each segment
    start_idx = 0
    
    # Iterate over large gaps to segment the data
    for idx in np.where(large_gaps)[0]:
        segment = df.iloc[start_idx:idx].copy()  # Copy the segment
        if len(segment) > 1:
            segment.set_index('datetime', inplace=True)
            segment_resampled = segment[columns_to_resample].resample(f'{1000/target_freq}L').mean().interpolate()
            resampled_segments.append(segment_resampled)
        start_idx = idx + 1
    
    # Handle the last segment after the final large gap
    segment = df.iloc[start_idx:].copy()
    if len(segment) > 1:
        segment.set_index('datetime', inplace=True)
        segment_resampled = segment[columns_to_resample].resample(f'{1000/target_freq}L').mean().interpolate()
        resampled_segments.append(segment_resampled)
    
    # Combine all resampled segments back together
    resampled_df = pd.concat(resampled_segments).reset_index()
    
    # Merge resampled data back with original non-resampled data (except tempObject)
    non_resampled_df = df.drop(columns=columns_to_resample + ['time_diff'], errors='ignore').reset_index(drop=True)
    merged_df = pd.merge_asof(resampled_df, non_resampled_df, on='datetime', direction='nearest')
    
    return merged_df

def resample_temp_object(df, target_freq, time_col='unixTimes'):
    df = df.copy()
    df['datetime'] = pd.to_datetime(df[time_col], unit='ms')
    df.set_index('datetime', inplace=True)
    
    # Resample the tempObject column along with unixTimes and sleep_label at its original frequency
    temp_resampled = df[['tempObject', time_col, 'sleep_label']].resample(f'{1000/target_freq}L').mean().interpolate()
    
    return temp_resampled.reset_index()

def process_each_file(file_path, config,verbos=False):
    window_size_seconds = config['windowing']['window_size_seconds']
    step_size_seconds = config['windowing']['step_size_seconds']
    
    # Access frequencies
    freq_acc = config['frequencies']['accelerometer']
    freq_gyro = config['frequencies']['gyroscope']
    freq_ppg = config['frequencies']['ppg']
    freq_temp = config['frequencies']['temperature']
    target_fs = 200
    # Calculate the number of samples in the window and step for each sensor type
    window_size_acc = int(window_size_seconds * freq_acc)
    step_size_acc = int(step_size_seconds * freq_acc)

    window_size_gyro = int(window_size_seconds * freq_gyro)
    step_size_gyro = int(step_size_seconds * freq_gyro)

    window_size_ppg = int(window_size_seconds * target_fs)
    step_size_ppg = int(step_size_seconds * target_fs)

    window_size_temp = int(window_size_seconds * freq_temp)
    step_size_temp = int(step_size_seconds * freq_temp)
    
    acc_lowcut = config['filters']['accelerometer']['lowcut']
    acc_highcut = config['filters']['accelerometer']['highcut']
    gyro_lowcut = config['filters']['gyroscope']['lowcut']
    gyro_highcut = config['filters']['gyroscope']['highcut']
    ppg_lowcut = config['filters']['ppg']['lowcut']
    ppg_highcut = config['filters']['ppg']['highcut']
    
    feature_matrix = []
    labels = []
    
    windowed_data = []
    total_steps=0

    df = pd.read_csv(file_path,usecols=["unixTimes","sleep_stage","accelerometerX", 
                                        "accelerometerY", "accelerometerZ","gyroscopeX", 
                                        "gyroscopeY", "gyroscopeZ","ledIR", "ledRed", 
                                        "ledGreen","tempObject" ],low_memory=False)

    # Create 'sleep_label' based on 'sleep_stage'
    df['sleep_label'] = df['sleep_stage'].apply(lambda x: 0 if x == 'WK' else 1)

    #df = filter_good_bad_segments(df, config)
    #df = filter_good_bad_segments_OLD(df, config)


    columns_to_resample = ["accelerometerX", "accelerometerY", "accelerometerZ", 
                           "gyroscopeX", "gyroscopeY", "gyroscopeZ", 
                           "ledIR", "ledRed", "ledGreen"]

    resampled_df = resample_data(df, freq_acc, columns_to_resample)

    # Handle `tempObject` separately at its original frequency
    temp_resampled_df = resample_temp_object(df, target_freq=freq_temp)



    del df
    gc.collect()

    acc_df = preprocess_sensor_data(resampled_df, config['features']['accelerometer'], 
                                    acc_lowcut, acc_highcut,freq_acc)
    gyro_df = preprocess_sensor_data(resampled_df, config['features']['gyroscope'], 
                                     gyro_lowcut, gyro_highcut,freq_gyro)
    ppg_df = preprocess_sensor_data_ppg(resampled_df, config['features']['ppg'],
                                    ppg_lowcut,ppg_highcut,freq_ppg,target_fs)
    temp_df = preprocess_sensor_data(temp_resampled_df, config['features']['temperature'])

    del resampled_df, temp_resampled_df
    gc.collect()


#                 calc_freq_acc = calculate_frequency(acc_df)
#                 calc_freq_gyro = calculate_frequency(gyro_df)
#                 calc_freq_ppg = calculate_frequency(ppg_df)
#                 calc_freq_temp = calculate_frequency(temp_df)

    if verbos:
        print(f"Calculated Accelerometer frequency: {calc_freq_acc} Hz")
        print(f"Calculated Gyroscope frequency: {calc_freq_gyro} Hz")
        print(f"Calculated PPG frequency: {calc_freq_ppg} Hz")
        print(f"Calculated Temp frequency: {calc_freq_temp} Hz")

        print(acc_df.shape)
        print(gyro_df.shape)
        print(ppg_df.shape)
        print(temp_df.shape)

        print("window_size_acc",window_size_acc)
        print("window_size_gyro",window_size_gyro)
        print("window_size_ppg",window_size_ppg)
        print("window_size_temp",window_size_temp)

    # Initialize starting indices for each sensor
    start_idx_acc = 0
    start_idx_gyro = 0
    start_idx_ppg = 0
    start_idx_temp = 0

    while (start_idx_acc + window_size_acc <= len(acc_df) and
           start_idx_gyro + window_size_gyro <= len(gyro_df) and
           start_idx_ppg + window_size_ppg <= len(ppg_df) and
           start_idx_temp + window_size_temp <= len(temp_df)):

        acc_window = np.expand_dims(acc_df[["accelerometerX", "accelerometerY", "accelerometerZ"]]
                                    .iloc[start_idx_acc:start_idx_acc + window_size_acc].values, axis=-1)
        gyro_window = np.expand_dims(gyro_df[["gyroscopeX", "gyroscopeY", "gyroscopeZ"]]
                                     .iloc[start_idx_gyro:start_idx_gyro + window_size_gyro].values, axis=-1)
        ppg_window = np.expand_dims(ppg_df[["ledIR", "ledRed", "ledGreen"]]
                                    .iloc[start_idx_ppg:start_idx_ppg + window_size_ppg].values, axis=-1)
        temp_window = np.expand_dims(temp_df[["tempObject"]]
                                     .iloc[start_idx_temp:start_idx_temp + window_size_temp].values, axis=-1)
        # Extract features for each PPG channel separately
        ppg_features = {}
        for i, channel_name in enumerate(["ledIR", "ledRed", "ledGreen"]):
            channel_data = ppg_window[:, i, 0]
            channel_features = extract_ppg_features(channel_data, target_fs,channel_name)

            # Add channel-specific names to features
            channel_features = {f"{channel_name}_{k}": v for k, v in channel_features.items()}
            ppg_features.update(channel_features)

        #print(ppg_features.keys())
        # For each windowed data segment
        gyro_features = extract_statistical_features(gyro_window, config['features']['gyroscope'])
        acc_features = extract_statistical_features(acc_window, config['features']['accelerometer'])
        temp_features = extract_statistical_features(temp_window, config['features']['temperature'])

        all_features = {**gyro_features, **acc_features, **temp_features, **ppg_features}
        #all_features = {**gyro_features, **acc_features, **temp_features}
        feature_vector = np.array(list(all_features.values()))  # Convert to 1D NumPy array

        # Assuming labels are binary and come from accelerometer data
        mean_label = np.mean(acc_df['sleep_label'].iloc[start_idx_acc:start_idx_acc + window_size_acc])
        label = 1.0 if mean_label > 0.5 else 0.0
        
        if not np.all(np.isfinite(feature_vector)):
            #print(f"Non-finite values detected in feature vector, replacing with NaN")
            feature_vector = np.nan_to_num(feature_vector, nan=np.nan, posinf=np.nan, neginf=np.nan)

            #print("Updated Feature Vector:", feature_vector)
        #print(feature_vector.shape)
        feature_matrix.append(feature_vector)
        labels.append(label)


        # Increment each sensor's index according to its own step size
        start_idx_acc += step_size_acc
        start_idx_gyro += step_size_gyro
        start_idx_ppg += step_size_ppg
        start_idx_temp += step_size_temp

        
    del acc_df, gyro_df, ppg_df, temp_df
     
    if verbos:
        print("Checking for non-finite values in feature matrix...")
        print("NaN values:", np.isnan(feature_matrix).any())
        print("Inf values:", np.isinf(feature_matrix).any())

    #print(feature_matrix)
    # Convert feature matrix and labels to NumPy arrays
    feature_matrix = np.array(feature_matrix)
    labels = np.array(labels)

    if verbos:
        # Check the structure of the feature matrix
        print("Feature matrix shape:", feature_matrix.shape)

#     # Apply imputation across the entire dataset
#     imputer = SimpleImputer(strategy='median')
#     imputed_features = imputer.fit_transform(feature_matrix)
    imputer = KNNImputer(n_neighbors=5, weights="uniform")  # You can adjust `n_neighbors`
    imputed_features = imputer.fit_transform(feature_matrix)
    print(imputed_features.shape)
    
    return imputed_features, labels
    
def create_tf_dataset(features, labels, batch_size=32, shuffle=True, buffer_size=1000):
    """
    Create a TensorFlow dataset from features and labels.
    Includes optional shuffling, batching, and prefetching.
    """
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)  # Shuffle the data
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def _load_all_files_as_dataset(directory_paths, config, batch_size=32, shuffle=True):
    """
    Load all files from directories and create a TensorFlow dataset.
    Includes shuffling within the dataset creation.
    """
    all_features = []
    all_labels = []

    # Iterate over all directories and files
    for dir_path in directory_paths:
        for filename in os.listdir(dir_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(dir_path, filename)
                features, labels = process_each_file(file_path, config)
                all_features.append(features)
                all_labels.append(labels)
            break
        break
    # Combine features and labels from all files
    all_features = np.vstack(all_features)
    all_labels = np.hstack(all_labels)

    # Create TensorFlow dataset
    dataset = create_tf_dataset(all_features, all_labels, batch_size=batch_size, shuffle=shuffle)
    return dataset

def buffered_data_generator(directory_paths, config, buffer_size=2000):
    """
    Generator that buffers data from multiple files, shuffles, and yields samples.
    """
    buffer = []

    for dir_path in directory_paths:
        for filename in os.listdir(dir_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(dir_path, filename)

                # Process the file to get features and labels
                features, labels = process_each_file(file_path, config)

                # Add to buffer
                for feature, label in zip(features, labels):
                    buffer.append((feature, label))

                    # If buffer is full, shuffle and yield samples
                    if len(buffer) >= buffer_size:
                        np.random.shuffle(buffer)
                        for sample in buffer:
                            yield sample
                        buffer = []  # Clear the buffer

    # Yield remaining samples in the buffer
    if buffer:
        np.random.shuffle(buffer)
        for sample in buffer:
            yield sample
def load_all_files_as_dataset(directory_paths, config, batch_size=32, shuffle=True, buffer_size=2000):
    """
    Load data from multiple files using a generator with buffering and shuffling.
    """
    def generator():
        yield from buffered_data_generator(directory_paths, config, buffer_size=buffer_size)

    output_types = (tf.float32, tf.float32)
    output_shapes = ((None,), ())

    dataset = tf.data.Dataset.from_generator(generator, output_types=output_types, output_shapes=output_shapes)
    dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(buffer_size)

    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


if __name__ == '__main__':
    # Load configuration
    config = load_config()
    random_seed = 100
    features_acc = config['features']['accelerometer']
    features_gyro = config['features']['gyroscope']
    features_ppg = config['features']['ppg']
    features_temp = config['features']['temperature']
    
    label_col = config['labels']['label_column']
    
    window_size_seconds = config['windowing']['window_size_seconds']
    step_size_seconds = config['windowing']['step_size_seconds']
    batch_size = config['windowing']['batch_size']
    
    
    
    # Directory containing the Excel files
    directory_path = ['./data/5folds/fold1/','./data/5folds/fold2/','./data/5folds/fold3/']
    train_dataset = load_all_files_as_dataset(directory_path, config, batch_size=32, shuffle=True, buffer_size=500)

    for batch_features, batch_labels in train_dataset.take(1):
        print("Batch features shape:", batch_features.shape)
        print("Batch labels shape:", batch_labels.shape)
        break
    
    
    
    
