import pandas as pd
import numpy as np
import os
import tensorflow as tf
import json
from collections import Counter 
from sklearn.utils import class_weight
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import RobustScaler,MinMaxScaler,StandardScaler
from scipy.interpolate import interp1d
import gc
import neurokit2 as nk

#FROM PROJECT FILES
from ppg_preprocess import filter_good_bad_segments

# FILTERS
# High-pass and band-pass filters
def butter_bandpass(lowcut, highcut, fs, order=5):
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

# PROCESS EACH SENSOR DATA

def preprocess_sensor_data(df, features, lowcut=None, highcut=None, fs=25,verbos=False):
    sensor_df = df[['unixTimes'] + features + ['sleep_label','sleep_stage'] ].dropna() 
    sensor_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    sensor_df.dropna(inplace=True)
    
    # Apply bandpass filter to each feature
    if lowcut!=None and highcut!=None:
        for feature in features:
            #sensor_df[feature] = remove_peaks_and_interpolate(sensor_df, feature)
            sensor_df[feature] = bandpass_filter(sensor_df[feature], lowcut, highcut, fs)
    elif lowcut!=None and highcut==None:
        for feature in features:
            #sensor_df[feature] = remove_peaks_and_interpolate(sensor_df, feature)
            sensor_df[feature] = highpass_filter(sensor_df[feature], lowcut, fs,order=5)
    
    #scaler = StandardScaler()
    scaler = RobustScaler()
    for feature in features:
        sensor_df[feature] = scaler.fit_transform(sensor_df[feature].values.reshape(-1, 1))
    return sensor_df

def preprocess_sensor_data_ppg(df, features, lowcut=None, highcut=None, fs=25,verbos=False):
    sensor_df = df[['unixTimes'] + features + ['sleep_label']].dropna()
    sensor_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    sensor_df.dropna(inplace=True)
    
    # Apply bandpass filter to each feature
    if lowcut!=None and highcut!=None:
        for feature in features:
            #sensor_df[feature] = remove_peaks_and_interpolate(sensor_df, feature)
            sensor_df[feature] = bandpass_filter(sensor_df[feature], lowcut, highcut, fs)  
    elif lowcut!=None and highcut==None:
        for feature in features:
            #sensor_df[feature] = remove_peaks_and_interpolate(sensor_df, feature)
            sensor_df[feature] = highpass_filter(sensor_df[feature], lowcut, fs,order=5)
    
    #scaler = StandardScaler()
    scaler = RobustScaler()
    for feature in features:
        #sensor_df[feature] = nk.ppg_clean(sensor_df[feature].values, sampling_rate=fs, method='elgendi')
        sensor_df[feature] = scaler.fit_transform(sensor_df[feature].values.reshape(-1, 1))
    return sensor_df

# Load configuration
def load_config(config_file='config.json'):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

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
#     temp_resampled = df[['tempObject', time_col, 'sleep_label','sleep_stage']].resample(f'{1000/target_freq}L').mean().interpolate()
    
    # Resample numeric columns and aggregate categorical separately
    temp_resampled = df.resample(f'{1000/target_freq}L').agg({
        'tempObject': 'mean',  # Resample tempObject with mean
        time_col: 'mean',  # Resample unixTimes with mean
        'sleep_label': 'mean',  # Resample sleep_label with mean
        'sleep_stage': lambda x: x.mode()[0] if not x.mode().empty else None  # Use mode for sleep_stage
    })

    
    return temp_resampled.reset_index()

def process_each_file(directory_path, config,verbos=False):
    window_size_seconds = config['windowing']['window_size_seconds']
    step_size_seconds = config['windowing']['step_size_seconds']
    
    # Access frequencies
    freq_acc = config['frequencies']['accelerometer']
    freq_gyro = config['frequencies']['gyroscope']
    freq_ppg = config['frequencies']['ppg']
    freq_temp = config['frequencies']['temperature']

    # Calculate the number of samples in the window and step for each sensor type
    window_size_acc = int(window_size_seconds * freq_acc)
    step_size_acc = int(step_size_seconds * freq_acc)

    window_size_gyro = int(window_size_seconds * freq_gyro)
    step_size_gyro = int(step_size_seconds * freq_gyro)

    window_size_ppg = int(window_size_seconds * freq_ppg)
    step_size_ppg = int(step_size_seconds * freq_ppg)

    window_size_temp = int(window_size_seconds * freq_temp)
    step_size_temp = int(step_size_seconds * freq_temp)
    
    acc_lowcut = config['filters']['accelerometer']['lowcut']
    acc_highcut = config['filters']['accelerometer']['highcut']
    gyro_lowcut = config['filters']['gyroscope']['lowcut']
    gyro_highcut = config['filters']['gyroscope']['highcut']
    ppg_lowcut = config['filters']['ppg']['lowcut']
    ppg_highcut = config['filters']['ppg']['highcut']

    windowed_data = []
    total_steps=0
    for dir_path in directory_path:
        for filename in os.listdir(dir_path):
            if verbos:
                print(filename)
            if filename.endswith(".csv"):
                file_path = os.path.join(dir_path, filename)
                df = pd.read_csv(file_path,usecols=["unixTimes","sleep_stage","accelerometerX", 
                                                    "accelerometerY", "accelerometerZ","gyroscopeX", 
                                                    "gyroscopeY", "gyroscopeZ","ledIR", "ledRed", 
                                                    "ledGreen","tempObject" ],low_memory=False)

                # Create 'sleep_label' based on 'sleep_stage'
                df = df[df['sleep_stage'] != 'NS']
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
                                                ppg_lowcut,ppg_highcut,freq_ppg)
                temp_df = preprocess_sensor_data(temp_resampled_df, config['features']['temperature'])

                del resampled_df, temp_resampled_df
                gc.collect()


                calc_freq_acc = calculate_frequency(acc_df)
                calc_freq_gyro = calculate_frequency(gyro_df)
                calc_freq_ppg = calculate_frequency(ppg_df)
                calc_freq_temp = calculate_frequency(temp_df)

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

                    # Split windows as in create_sensor_generator
                    acc_window = np.expand_dims(acc_df[["accelerometerX", "accelerometerY", "accelerometerZ"]]
                                                .iloc[start_idx_acc:start_idx_acc + window_size_acc].values, axis=-1)
                    gyro_window = np.expand_dims(gyro_df[["gyroscopeX", "gyroscopeY", "gyroscopeZ"]]
                                                 .iloc[start_idx_gyro:start_idx_gyro + window_size_gyro].values, axis=-1)
                    ppg_window = np.expand_dims(ppg_df[["ledIR", "ledRed", "ledGreen"]]
                                                .iloc[start_idx_ppg:start_idx_ppg + window_size_ppg].values, axis=-1)
                    temp_window = np.expand_dims(temp_df[["tempObject"]]
                                                 .iloc[start_idx_temp:start_idx_temp + window_size_temp].values, axis=-1)

                    # Assuming labels are binary and come from accelerometer data
                    mean_label = np.mean(acc_df['sleep_label'].iloc[start_idx_acc:start_idx_acc + window_size_acc])
                    label = 1.0 if mean_label > 0.5 else 0.0

                    # Yield the current window and the corresponding label
                    yield (acc_window, gyro_window, ppg_window, temp_window), label

                    # Increment each sensor's index according to its own step size
                    start_idx_acc += step_size_acc
                    start_idx_gyro += step_size_gyro
                    start_idx_ppg += step_size_ppg
                    start_idx_temp += step_size_temp

                del acc_df, gyro_df, ppg_df, temp_df
                gc.collect()  # Explicitly trigger garbage collection after each file
                
def process_per_file(file_path, config,verbos=False):
    window_size_seconds = config['windowing']['window_size_seconds']
    step_size_seconds = config['windowing']['step_size_seconds']
    
    # Access frequencies
    freq_acc = config['frequencies']['accelerometer']
    freq_gyro = config['frequencies']['gyroscope']
    freq_ppg = config['frequencies']['ppg']
    freq_temp = config['frequencies']['temperature']

    # Calculate the number of samples in the window and step for each sensor type
    window_size_acc = int(window_size_seconds * freq_acc)
    step_size_acc = int(step_size_seconds * freq_acc)

    window_size_gyro = int(window_size_seconds * freq_gyro)
    step_size_gyro = int(step_size_seconds * freq_gyro)

    window_size_ppg = int(window_size_seconds * freq_ppg)
    step_size_ppg = int(step_size_seconds * freq_ppg)

    window_size_temp = int(window_size_seconds * freq_temp)
    step_size_temp = int(step_size_seconds * freq_temp)
    
    acc_lowcut = config['filters']['accelerometer']['lowcut']
    acc_highcut = config['filters']['accelerometer']['highcut']
    gyro_lowcut = config['filters']['gyroscope']['lowcut']
    gyro_highcut = config['filters']['gyroscope']['highcut']
    ppg_lowcut = config['filters']['ppg']['lowcut']
    ppg_highcut = config['filters']['ppg']['highcut']

    df = pd.read_csv(file_path,usecols=["unixTimes","sleep_stage","accelerometerX", 
                                        "accelerometerY", "accelerometerZ","gyroscopeX", 
                                        "gyroscopeY", "gyroscopeZ","ledIR", "ledRed", 
                                        "ledGreen","tempObject" ],low_memory=False)

    # Create 'sleep_label' based on 'sleep_stage'
    df = df[df['sleep_stage'] != 'NS']
    df['sleep_label'] = df['sleep_stage'].apply(lambda x: 0 if x == 'WK' else 1)
    
    sleep_stage_distribution = df['sleep_stage'].value_counts()
    print("\nDistribution of Sleep Stages:")
    print(sleep_stage_distribution)
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
                                    ppg_lowcut,ppg_highcut,freq_ppg)
    temp_df = preprocess_sensor_data(temp_resampled_df, config['features']['temperature'])

    del resampled_df, temp_resampled_df
    gc.collect()
    
    # Initialize starting indices for each sensor
    start_idx_acc = 0
    start_idx_gyro = 0
    start_idx_ppg = 0
    start_idx_temp = 0

    while (start_idx_acc + window_size_acc <= len(acc_df) and
           start_idx_gyro + window_size_gyro <= len(gyro_df) and
           start_idx_ppg + window_size_ppg <= len(ppg_df) and
           start_idx_temp + window_size_temp <= len(temp_df)):

        # Split windows as in create_sensor_generator
        acc_window = np.expand_dims(acc_df[["accelerometerX", "accelerometerY", "accelerometerZ"]]
                                    .iloc[start_idx_acc:start_idx_acc + window_size_acc].values, axis=-1)
        gyro_window = np.expand_dims(gyro_df[["gyroscopeX", "gyroscopeY", "gyroscopeZ"]]
                                     .iloc[start_idx_gyro:start_idx_gyro + window_size_gyro].values, axis=-1)
        ppg_window = np.expand_dims(ppg_df[["ledIR", "ledRed", "ledGreen"]]
                                    .iloc[start_idx_ppg:start_idx_ppg + window_size_ppg].values, axis=-1)
        temp_window = np.expand_dims(temp_df[["tempObject"]]
                                     .iloc[start_idx_temp:start_idx_temp + window_size_temp].values, axis=-1)

        all_nan = acc_df['sleep_stage'].iloc[start_idx_acc:start_idx_acc + window_size_acc].isna().all()
        if all_nan:
            print("All NAN")
        else:
            # Assuming labels are binary and come from accelerometer data
            mean_label = np.mean(acc_df['sleep_label'].iloc[start_idx_acc:start_idx_acc + window_size_acc])
            label = 1.0 if mean_label > 0.5 else 0.0

            # Yield the current window and the corresponding label
            yield (
                (acc_window, gyro_window, ppg_window, temp_window),  # Features
                label,                                              # Aggregated label
                acc_df['sleep_label'].iloc[start_idx_acc:start_idx_acc + window_size_acc],
                acc_df['sleep_stage'].iloc[start_idx_acc:start_idx_acc + window_size_acc],
                (start_idx_acc,start_idx_acc + window_size_acc)# Whole window labels
            )

        # Increment each sensor's index according to its own step size
        start_idx_acc += step_size_acc
        start_idx_gyro += step_size_gyro
        start_idx_ppg += step_size_ppg
        start_idx_temp += step_size_temp

    del acc_df, gyro_df, ppg_df, temp_df
    gc.collect()  # Explicitly trigger garbage collection after each file
    
def process_each_file_StackedModel(directory_path, config,verbos=False):
    window_size_seconds = config['windowing']['window_size_seconds']
    step_size_seconds = config['windowing']['step_size_seconds']
    
    # Access frequencies
    freq_acc = config['frequencies']['accelerometer']
    freq_gyro = config['frequencies']['gyroscope']
    freq_ppg = config['frequencies']['ppg']
    freq_temp = config['frequencies']['temperature']

    # Calculate the number of samples in the window and step for each sensor type
    window_size_acc = int(window_size_seconds * freq_acc)
    step_size_acc = int(step_size_seconds * freq_acc)

    window_size_gyro = int(window_size_seconds * freq_gyro)
    step_size_gyro = int(step_size_seconds * freq_gyro)

    window_size_ppg = int(window_size_seconds * freq_ppg)
    step_size_ppg = int(step_size_seconds * freq_ppg)

    window_size_temp = int(window_size_seconds * freq_temp)
    step_size_temp = int(step_size_seconds * freq_temp)
    
    acc_lowcut = config['filters']['accelerometer']['lowcut']
    acc_highcut = config['filters']['accelerometer']['highcut']
    gyro_lowcut = config['filters']['gyroscope']['lowcut']
    gyro_highcut = config['filters']['gyroscope']['highcut']
    ppg_lowcut = config['filters']['ppg']['lowcut']
    ppg_highcut = config['filters']['ppg']['highcut']

    windowed_data = []
    total_steps=0
    for dir_path in directory_path:
        for filename in os.listdir(dir_path):
            if verbos:
                print(filename)
            if filename.endswith(".csv"):
                file_path = os.path.join(dir_path, filename)
                df = pd.read_csv(file_path,usecols=["unixTimes","sleep_stage","accelerometerX", 
                                                    "accelerometerY", "accelerometerZ","gyroscopeX", 
                                                    "gyroscopeY", "gyroscopeZ","ledIR", "ledRed", 
                                                    "ledGreen","tempObject" ],low_memory=False)

                df = df[:len(df) // 4]
                # Create 'sleep_label' based on 'sleep_stage'
                df = df[df['sleep_stage'] != 'NS']
                df['sleep_label'] = df['sleep_stage'].apply(lambda x: 0 if x == 'WK' else 1)
                
                
                columns_to_resample = ["accelerometerX", "accelerometerY", "accelerometerZ", 
                                       "gyroscopeX", "gyroscopeY", "gyroscopeZ", 
                                       "ledIR", "ledRed", "ledGreen"]

                resampled_df = resample_data(df, freq_acc, columns_to_resample)
                # Handle `tempObject` separately at its original frequency

                
                del df
                gc.collect()
                
                acc_df = preprocess_sensor_data(resampled_df, config['features']['accelerometer'], 
                                                acc_lowcut, acc_highcut,freq_acc)
                gyro_df = preprocess_sensor_data(resampled_df, config['features']['gyroscope'], 
                                                 gyro_lowcut, gyro_highcut,freq_gyro)
                ppg_df = preprocess_sensor_data_ppg(resampled_df, config['features']['ppg'],
                                                ppg_lowcut,ppg_highcut,freq_ppg)
                
                del resampled_df
                gc.collect()

                
                calc_freq_acc = calculate_frequency(acc_df)
                calc_freq_gyro = calculate_frequency(gyro_df)
                calc_freq_ppg = calculate_frequency(ppg_df)

                if verbos:
                    print(acc_df)
                    print(temp_df)
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
                       start_idx_ppg + window_size_ppg <= len(ppg_df)):

                    # Split windows as in create_sensor_generator
                    acc_window = np.expand_dims(acc_df[["accelerometerX", "accelerometerY", "accelerometerZ"]]
                                                .iloc[start_idx_acc:start_idx_acc + window_size_acc].values, axis=-1)
                    gyro_window = np.expand_dims(gyro_df[["gyroscopeX", "gyroscopeY", "gyroscopeZ"]]
                                                 .iloc[start_idx_gyro:start_idx_gyro + window_size_gyro].values, axis=-1)
                    ppg_window = np.expand_dims(ppg_df[["ledIR", "ledRed", "ledGreen"]]
                                                .iloc[start_idx_ppg:start_idx_ppg + window_size_ppg].values, axis=-1)
                    

                    # Assuming labels are binary and come from accelerometer data
                    mean_label = np.mean(acc_df['sleep_label'].iloc[start_idx_acc:start_idx_acc + window_size_acc])
                    label = 1.0 if mean_label > 0.5 else 0.0
                    
                    stacked_data = np.concatenate([acc_window, gyro_window, ppg_window], axis=-1)  # Shape: (1500, 3, 3)

                    # Yield the current window and the corresponding label
                    yield stacked_data, label

                    # Increment each sensor's index according to its own step size
                    start_idx_acc += step_size_acc
                    start_idx_gyro += step_size_gyro
                    start_idx_ppg += step_size_ppg
                    start_idx_temp += step_size_temp

                del acc_df, gyro_df, ppg_df
                gc.collect()  # Explicitly trigger garbage collection after each file

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
    batch_size = 1#config['windowing']['batch_size']
    
    
    
    # Directory containing the Excel files
    #directory_path = ['./data/5folds/fold3/','./data/5folds/fold2/','./data/5folds/fold1/']
    directory_path = ['./data/5folds/fold5/']
    
    # Prepare the dataset
#     train_dataset = tf.data.Dataset.from_generator(
#         lambda: process_each_file_StackedModel(directory_path, config),
#         output_types=((tf.float32, tf.float32, tf.float32, tf.float32), tf.int64),
#         output_shapes=(((None, None, 1), (None, None, 1), (None, None, 1), (None, None, 1)), ())
#     )

    train_dataset = tf.data.Dataset.from_generator(
        lambda: process_each_file_StackedModel(directory_path, config),
        output_types=( tf.float32, tf.int64),
        output_shapes=((None, None, 3), ())
    )
    
    # Shuffle, batch, and prefetch
    train_dataset = train_dataset.batch(batch_size).shuffle(800).prefetch(tf.data.experimental.AUTOTUNE)

    all_labels = []
    
    for x,y in train_dataset:
        print(x.shape)
        all_labels.extend(y)
        break
        
    all_labels = np.array(all_labels)
    print(len(all_labels))

    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
    class_weight_dict = dict(enumerate(class_weights))
    print("Class Weights:", class_weight_dict)





    
    
    
    
