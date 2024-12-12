import pandas as pd
import numpy as np
import os
import tensorflow as tf
import json
from collections import Counter 
from sklearn.utils import class_weight
from scipy.signal import butter, filtfilt
from ppg_preprocess import filter_good_bad_segments,filter_good_bad_segments_OLD
from sklearn.preprocessing import RobustScaler,MinMaxScaler,StandardScaler
from scipy.interpolate import interp1d
import gc
#from sklearn.metrics import classification_report

# Load configuration
def load_config(config_file='config.json'):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

# Read and preprocess data
def read_and_preprocess_files(directory_path):
    df_list = []

    # Read each Excel file and append to the list
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path)
            df_list.append(df)

    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(df_list, ignore_index=True)

    # Create a new column 'sleep_label' based on 'sleep_stage'
    combined_df['sleep_label'] = combined_df['sleep_stage'].apply(lambda x: 0 if x == 'WK' else 1)  # 0: awake, 1: sleep

    return combined_df

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_highpass(cutoff, fs, order=5):
    """
    Create a highpass filter.
    
    Args:
    cutoff (float): The cutoff frequency for the highpass filter.
    fs (float): The sampling frequency of the signal.
    order (int): The order of the filter. Higher values mean a sharper cutoff.
    
    Returns:
    b, a: Filter coefficients.
    """
    nyquist = 0.5 * fs
    high = cutoff / nyquist
    b, a = butter(order, high, btype='highpass')
    return b, a

def highpass_filter(data, cutoff, fs, order=5):
    """
    Apply a highpass filter to the data.
    
    Args:
    data (array-like): The input signal.
    cutoff (float): The cutoff frequency for the highpass filter.
    fs (float): The sampling frequency of the signal.
    order (int): The order of the filter. Higher values mean a sharper cutoff.
    
    Returns:
    y: The filtered signal.
    """
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def remove_peaks_and_interpolate(sensor_df, feature, lower_percentile=0.01, upper_percentile=0.99):
    """
    Remove high and low peaks based on quartiles and interpolate those values.
    
    Args:
    sensor_df (pd.DataFrame): The sensor data DataFrame.
    feature (str): The feature/column to apply peak removal and interpolation.
    lower_percentile (float): The lower percentile threshold for detecting low peaks (default 5%).
    upper_percentile (float): The upper percentile threshold for detecting high peaks (default 95%).

    Returns:
    pd.Series: The modified feature column after peak removal and interpolation.
    """
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
def _remove_peaks_and_interpolate(sensor_df, feature, lower_percentile=0.01, upper_percentile=0.99):
    lower_bound = sensor_df[feature].quantile(lower_percentile)
    upper_bound = sensor_df[feature].quantile(upper_percentile)

    clean_data = sensor_df[feature].clip(lower=lower_bound, upper=upper_bound)

    return clean_data.interpolate()

def preprocess_sensor_data(df, features, lowcut=None, highcut=None, fs=25):
    sensor_df = df[['unixTimes'] + features + ['sleep_label']].dropna()
    sensor_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    sensor_df.dropna(inplace=True)
    
    # Apply bandpass filter to each feature
    if lowcut!=None and highcut!=None:
        print("Bandpass")
        #scaler = StandardScaler()
        #scaler = RobustScaler()
        for feature in features:
            sensor_df[feature] = remove_peaks_and_interpolate(sensor_df, feature)
            sensor_df[feature] = bandpass_filter(sensor_df[feature], lowcut, highcut, fs)
            #sensor_df[feature] = scaler.fit_transform(sensor_df[feature].values.reshape(-1, 1))
    elif lowcut!=None and highcut==None:
        print("Highpass")
        #scaler = StandardScaler()
        #scaler = RobustScaler()
        for feature in features:
            sensor_df[feature] = remove_peaks_and_interpolate(sensor_df, feature)
            sensor_df[feature] = highpass_filter(sensor_df[feature], lowcut, fs,order=5)
            #sensor_df[feature] = scaler.fit_transform(sensor_df[feature].values.reshape(-1, 1))
    else:
        print("No frequency filter in data pipeline")
    
    scaler = StandardScaler()
    #scaler = RobustScaler()
    for feature in features:
        sensor_df[feature] = scaler.fit_transform(sensor_df[feature].values.reshape(-1, 1))
    return sensor_df


# Calculate the frequency of data collection
def calculate_frequency(data):
    time_diffs = np.diff(data['unixTimes'].values)  # Calculate time differences between successive samples
    print("Time differences (first 10):", time_diffs[:10])  # Debug: print first 10 time differences
    avg_time_diff = np.mean(time_diffs)  # Average time difference
    print("Average time difference:", avg_time_diff)  # Debug: print average time difference
    frequency = 1000 / avg_time_diff  # Frequency is the inverse of the average time difference
    return frequency

# PPG PREPROCESSING
def normalize_signal(data):
    return (data - np.mean(data)) / np.std(data)
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Create TensorFlow dataset for resampled data
def create_tf_dataset_resampled(data, features, label_col, window_size_seconds):
    feature_data = data[features].values
    label_data = data[label_col].values
    unix_times = data['unixTimes'].values
    timestamps = pd.to_datetime(unix_times, unit='ms')

    # Create a DataFrame with the features and labels
    df = pd.DataFrame(feature_data, columns=features)
    df['label'] = label_data
    df.index = timestamps

    # Resample to 30-second windows
    resampled = df.resample(f'{window_size_seconds}S')
    
    def gen():
        for _, window in resampled:
            if len(window) == window_size_seconds:  # Ensure window is the correct size
                yield window[features].values, window['label'].max()

    return tf.data.Dataset.from_generator(
        gen,
        output_types=(tf.float32, tf.int64),
        output_shapes=((window_size_seconds, len(features)), ())
    )

# Create generator functions for each sensor type
def create_sensor_generator(data, features, label_col, window_size, step_size):
    feature_data = data[features].values
    label_data = data[label_col].values
    
    def gen():
        start_idx = 0
        while start_idx + window_size <= len(data):
            window_indices = np.arange(start_idx, start_idx + window_size)
            #yield np.expand_dims(feature_data[window_indices], axis=-1), label_data[window_indices].max()
            #yield np.expand_dims(feature_data[window_indices], axis=-1), label_data[window_indices].max()
            
            features = np.expand_dims(feature_data[window_indices], axis=-1)
            mean_label = np.mean(label_data[window_indices])
            label = 1 if mean_label > 0.5 else 0
            yield features, label
            #yield np.expand_dims(feature_data[window_indices], axis=-1), np.mean(label_data[window_indices]) 
            start_idx += step_size
    return gen



def _create_sensor_generator_3(data, features, label_col, window_size, step_size, threshold=0.6):
    """
    Create a generator for sensor data with a labeling threshold of 75%.
    
    Parameters:
    - data: The DataFrame containing the sensor data.
    - features: List of feature column names.
    - label_col: Name of the label column.
    - window_size: Size of the window for splitting the data.
    - step_size: Step size for the sliding window.
    - threshold: Threshold for labeling (default is 75%).
    
    Returns:
    - A generator that yields features and labels.
    """
    feature_data = data[features].values
    label_data = data[label_col].values
    
    def gen():
        start_idx = 0
        while start_idx + window_size <= len(data):
            window_indices = np.arange(start_idx, start_idx + window_size)
            features_window = np.expand_dims(feature_data[window_indices], axis=-1)
            label_window = label_data[window_indices]

            # Apply the 75% threshold logic
            proportion_ones = np.mean(label_window)
            if proportion_ones >= threshold:
                label = 1
            elif proportion_ones <= (1 - threshold):
                label = 0
            else:
                label = 2 

            yield features_window, label
            start_idx += step_size
    
    return gen

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

def calculate_label_distribution(df, label_col):
    label_counts = df[label_col].value_counts()
    label_percentages = df[label_col].value_counts(normalize=True) * 100
    return label_counts, label_percentages

def calculate_class_weights(df, label_col):
    labels = df[label_col].values
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    return dict(enumerate(class_weights))

def test_time(df):
    df['datetime'] = pd.to_datetime(df['unixTimes'], unit='ms')

    # Calculate the differences between consecutive timestamps
    df['time_diff'] = df['datetime'].diff().dt.total_seconds()

    # Check for any big jumps
    big_jumps = df[df['time_diff'] > 1]  # Adjust the threshold as needed

    # Analyze datetime ranges
    start_time = df['datetime'].min()
    end_time = df['datetime'].max()

    print(f"Start time: {start_time}")
    print(f"End time: {end_time}")
    print(f"Total duration: {end_time - start_time}")

    print("Big jumps in timestamps:")
    print(big_jumps[['datetime', 'time_diff']])

    # Optional: Display the DataFrame with calculated time differences
    print(df.head())
    
def split_into_chunks(df, chunk_duration_minutes=5):
    df['datetime'] = pd.to_datetime(df['unixTimes'], unit='ms')
    df.set_index('datetime', inplace=True)
    chunks = [chunk for _, chunk in df.groupby(pd.Grouper(freq=f'{chunk_duration_minutes}T'))]
    return chunks


def resample_data(df, target_freq, columns_to_resample, time_col='unixTimes', max_gap=1*60000):  # max_gap in milliseconds
    df = df.copy()
    df['datetime'] = pd.to_datetime(df[time_col], unit='ms')
    
    # Identify large gaps
    df['time_diff'] = df['datetime'].diff().dt.total_seconds() * 1000  # in milliseconds
    large_gaps = df['time_diff'] > max_gap
    
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

def process_each_file(directory_path, config):
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
    ppg_highcut = None#config['filters']['ppg']['highcut']

    windowed_data = []
    total_steps=0
    
    for filename in os.listdir(directory_path):
        print(filename)
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path)
            
            # Create 'sleep_label' based on 'sleep_stage'
            df['sleep_label'] = df['sleep_stage'].apply(lambda x: 0 if x == 'WK' else 1)
            
            df = filter_good_bad_segments(df, config)
            #df = filter_good_bad_segments_OLD(df, config)
            
        
            columns_to_resample = ["accelerometerX", "accelerometerY", "accelerometerZ", 
                                   "gyroscopeX", "gyroscopeY", "gyroscopeZ", 
                                   "ledIR", "ledRed", "ledGreen"]

            resampled_df = resample_data(df, freq_acc, columns_to_resample)
            print("resampled_df",resampled_df.shape)
            # Handle `tempObject` separately at its original frequency
            temp_resampled_df = resample_temp_object(df, target_freq=freq_temp)
            print("temp_resampled_df",temp_resampled_df.shape)
            

            del df
            gc.collect()
            
            acc_df = preprocess_sensor_data(resampled_df, config['features']['accelerometer'], 
                                            acc_lowcut, acc_highcut,freq_acc)
            gyro_df = preprocess_sensor_data(resampled_df, config['features']['gyroscope'], 
                                             gyro_lowcut, gyro_highcut,freq_gyro)
            ppg_df = preprocess_sensor_data(resampled_df, config['features']['ppg'],
                                            ppg_lowcut,ppg_highcut,freq_ppg)
            temp_df = preprocess_sensor_data(temp_resampled_df, config['features']['temperature'])

            del resampled_df, temp_resampled_df
            gc.collect()
        
            # Preprocess sensor data for each file
#             acc_df = preprocess_sensor_data_acc(resampled_df, config['features']['accelerometer'], 
#                                             acc_lowcut, acc_highcut,freq_acc)
#             gyro_df = preprocess_sensor_data_gyro(resampled_df, config['features']['gyroscope'], 
#                                              gyro_lowcut, gyro_highcut,freq_gyro)
#             ppg_df = preprocess_sensor_data_ppg(resampled_df, config['features']['ppg'],
#                                             ppg_lowcut,ppg_highcut,freq_ppg)
#             temp_df = preprocess_sensor_data_temp(temp_resampled_df, config['features']['temperature'])
            
            calc_freq_acc = calculate_frequency(acc_df)
            calc_freq_gyro = calculate_frequency(gyro_df)
            calc_freq_ppg = calculate_frequency(ppg_df)
            calc_freq_temp = calculate_frequency(temp_df)
            
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

            # Create generators for each sensor type
            acc_gen = create_sensor_generator(acc_df, 
                                              config['features']['accelerometer'], 
                                              config['labels']['label_column'], 
                                              window_size_acc, step_size_acc)
            
            gyro_gen = create_sensor_generator(gyro_df, 
                                               config['features']['gyroscope'], 
                                               config['labels']['label_column'], 
                                               window_size_gyro, step_size_gyro)
            
            ppg_gen = create_sensor_generator(ppg_df, 
                                              config['features']['ppg'], 
                                              config['labels']['label_column'], 
                                              window_size_ppg, step_size_ppg)
            
            temp_gen = create_sensor_generator(temp_df, config['features']['temperature'], 
                                               config['labels']['label_column'], 
                                               window_size_temp, step_size_temp)

            # Store each generator output for further processing
            windowed_data.append((acc_gen, gyro_gen, ppg_gen, temp_gen))
            #windowed_data.append((acc_gen, gyro_gen, ppg_gen))
            steps_per_file = (len(acc_df) - window_size_acc) // step_size_acc + 1
            total_steps += steps_per_file
        
            
    ppg_input_shape = (window_size_ppg, len(config['features']['ppg']), 1)
    gyro_input_shape = (window_size_gyro, len(config['features']['gyroscope']), 1)
    acc_input_shape = (window_size_acc, len(config['features']['accelerometer']), 1)
    temp_input_shape = (window_size_temp, len(config['features']['temperature']), 1)
    
    input_shapes = [ppg_input_shape, gyro_input_shape, acc_input_shape,temp_input_shape]
    
    return windowed_data, total_steps, input_shapes

def process_and_create_datasets(windowed_data, config):
    
    
    datasets = []

    # Convert window sizes and step sizes to integers
    window_size_acc = int(config['windowing']['window_size_seconds'] * config['frequencies']['accelerometer'])
    window_size_gyro = int(config['windowing']['window_size_seconds'] * config['frequencies']['gyroscope'])
    window_size_ppg = int(config['windowing']['window_size_seconds'] * config['frequencies']['ppg'])
    window_size_temp = int(config['windowing']['window_size_seconds'] * config['frequencies']['temperature'])
    
    output_types = ((tf.float32, tf.float32, tf.float32, tf.float32), tf.int64)
    output_shapes = (
        (
            (window_size_acc, len(config['features']['accelerometer']), 1), 
            (window_size_gyro, len(config['features']['gyroscope']), 1), 
            (window_size_ppg, len(config['features']['ppg']), 1), 
            (window_size_temp, len(config['features']['temperature']), 1)
        ), 
        ()
    )
#     output_types = ((tf.float32, tf.float32, tf.float32), tf.int64)
#     output_shapes = (
#         (
#             (window_size_acc, len(config['features']['accelerometer']), 1), 
#             (window_size_gyro, len(config['features']['gyroscope']), 1), 
#             (window_size_ppg, len(config['features']['ppg']), 1)
#         ), 
#         ()
#     )
    
    for (acc_gen, gyro_gen, ppg_gen, temp_gen) in windowed_data:
        combined_dataset = create_combined_tf_dataset([acc_gen, gyro_gen, ppg_gen, temp_gen], output_types, output_shapes)
        datasets.append(combined_dataset)
        
#     for (acc_gen, gyro_gen, ppg_gen) in windowed_data:
#         combined_dataset = create_combined_tf_dataset([acc_gen, gyro_gen, ppg_gen], output_types, output_shapes)
#         datasets.append(combined_dataset)
    
    if datasets:
        final_dataset = datasets[0]
        for ds in datasets[1:]:
            final_dataset = final_dataset.concatenate(ds)
        return final_dataset
    else:
        return None

import pandas as pd
import numpy as np


def calculate_min_max_values(dataset):
    """
    Calculate the min and max values for each sensor in the dataset.

    Parameters:
    - dataset: The dataset containing sensor data.

    Returns:
    - A dictionary containing min and max values for each sensor.
    """
    min_max_values = {
        'sensor_1': {'min': float('inf'), 'max': float('-inf')},
        'sensor_2': {'min': float('inf'), 'max': float('-inf')},
        'sensor_3': {'min': float('inf'), 'max': float('-inf')},
        'sensor_4': {'min': float('inf'), 'max': float('-inf')}
    }

    for data, label in dataset:
        sensor_1_data, sensor_2_data, sensor_3_data, sensor_4_data = data

        # Calculate min and max for sensor 1
        min_max_values['sensor_1']['min'] = min(min_max_values['sensor_1']['min'], tf.reduce_min(sensor_1_data).numpy())
        min_max_values['sensor_1']['max'] = max(min_max_values['sensor_1']['max'], tf.reduce_max(sensor_1_data).numpy())

        # Calculate min and max for sensor 2
        min_max_values['sensor_2']['min'] = min(min_max_values['sensor_2']['min'], tf.reduce_min(sensor_2_data).numpy())
        min_max_values['sensor_2']['max'] = max(min_max_values['sensor_2']['max'], tf.reduce_max(sensor_2_data).numpy())

        # Calculate min and max for sensor 3
        min_max_values['sensor_3']['min'] = min(min_max_values['sensor_3']['min'], tf.reduce_min(sensor_3_data).numpy())
        min_max_values['sensor_3']['max'] = max(min_max_values['sensor_3']['max'], tf.reduce_max(sensor_3_data).numpy())
        
        min_max_values['sensor_4']['min'] = min(min_max_values['sensor_4']['min'], tf.reduce_min(sensor_4_data).numpy())
        min_max_values['sensor_4']['max'] = max(min_max_values['sensor_4']['max'], tf.reduce_max(sensor_4_data).numpy())

    return min_max_values
    
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
    directory_path = './data/test'
    
    windowed_data, total_steps, input_shapes = process_each_file(directory_path, config)
    
    print("total_steps",total_steps)
    print("input_shapes",input_shapes)
    # Create combined datasets for all files
    final_dataset = process_and_create_datasets(windowed_data, config)

    
    count=0
    for data in final_dataset:
        #print(data)
        features, label = data
        print(features[0])
        #for i, feature_set in enumerate(features):
            #print(f"Features shape for sensor {i+1}:", feature_set.numpy().shape)
        #print("Label:", label.numpy())
        count=count+1
        break
        
    print("total count",count)

    all_labels = []
    for _, label in final_dataset:
        all_labels.append(label.numpy())
    all_labels = np.array(all_labels)
    
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
    class_weight_dict = dict(enumerate(class_weights))
    print("Class Weights:", class_weight_dict)









    
    
    
    
