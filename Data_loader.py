import pandas as pd
import numpy as np
import os
import tensorflow as tf
import json
from collections import Counter 
from sklearn.utils import class_weight
from scipy.signal import butter, filtfilt
from ppg_preprocess import filter_good_bad_segments
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

# Separate and preprocess sensor data
def _preprocess_sensor_data(df, features):
    sensor_df = df[['unixTimes'] + features + ['sleep_label']].dropna()
    # Check for NaNs or infinite values and remove them
    sensor_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    sensor_df.dropna(inplace=True)
    # Normalize the data
    feature_data = sensor_df[features].values
    mean = np.mean(feature_data, axis=0)
    std = np.std(feature_data, axis=0)
    sensor_df[features] = (feature_data - mean) / std
    sensor_df[features] = feature_data 
    return sensor_df

def preprocess_sensor_data(df, features, lowcut=None, highcut=None, fs=25):
    sensor_df = df[['unixTimes'] + features + ['sleep_label']].dropna()
    sensor_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    sensor_df.dropna(inplace=True)

    # Apply bandpass filter to each feature
    if lowcut!=None and highcut!=None:
        for feature in features:
            sensor_df[feature] = bandpass_filter(sensor_df[feature], lowcut, highcut, fs)
    
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
def _resample_data(df, target_freq, time_col='unixTimes'):
    df = df.copy()
    df['datetime'] = pd.to_datetime(df[time_col], unit='ms')
    df.set_index('datetime', inplace=True)
    df = df.resample(f'{1000/target_freq}L').mean().interpolate()
    df.reset_index(drop=True, inplace=True)
    return df

def __resample_data(df, target_freq, time_col='unixTimes', max_gap=1*60000):  # max_gap in milliseconds
    df = df.copy()
    df['datetime'] = pd.to_datetime(df[time_col], unit='ms')
    
    # Identify large gaps
    df['time_diff'] = df['datetime'].diff().dt.total_seconds() * 1000  # in milliseconds
    large_gaps = df['time_diff'] > max_gap
    
    print(sum(large_gaps))
    
    # Initialize an empty list to collect resampled segments
    resampled_segments = []
    
    # Start index for each segment
    start_idx = 0
    
    # Iterate over large gaps to segment the data
    for idx in np.where(large_gaps)[0]:
        segment = df.iloc[start_idx:idx].copy()  # Copy the segment
        if len(segment) > 1:
            segment.set_index('datetime', inplace=True)
            segment_resampled = segment.resample(f'{1000/target_freq}L').mean().interpolate()
            resampled_segments.append(segment_resampled)
        start_idx = idx + 1
    
    # Handle the last segment after the final large gap
    segment = df.iloc[start_idx:].copy()
    if len(segment) > 1:
        segment.set_index('datetime', inplace=True)
        segment_resampled = segment.resample(f'{1000/target_freq}L').mean().interpolate()
        resampled_segments.append(segment_resampled)
    
    # Combine all resampled segments back together
    resampled_df = pd.concat(resampled_segments).reset_index(drop=True)
    
    # Drop the auxiliary columns
    resampled_df.drop(columns=['time_diff'], inplace=True, errors='ignore')
    
    return resampled_df

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
    ppg_highcut = config['filters']['ppg']['highcut']

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
            
            columns_to_resample = ["accelerometerX", "accelerometerY", "accelerometerZ", 
                                   "gyroscopeX", "gyroscopeY", "gyroscopeZ", 
                                   "ledIR", "ledRed", "ledGreen"]

            resampled_df = resample_data(df, freq_acc, columns_to_resample)
            print("resampled_df",resampled_df.shape)
            # Handle `tempObject` separately at its original frequency
            temp_resampled_df = resample_temp_object(df, target_freq=freq_temp)
            print("temp_resampled_df",temp_resampled_df.shape)
            
            

            #df = df.dropna(subset=config['features']['accelerometer']+config['features']['gyroscope']+config['features']['ppg'])

            
            # Preprocess sensor data for each file
            acc_df = preprocess_sensor_data(resampled_df, config['features']['accelerometer'], acc_lowcut, acc_highcut,freq_acc)
            gyro_df = preprocess_sensor_data(resampled_df, config['features']['gyroscope'], gyro_lowcut, gyro_highcut,freq_gyro)
            ppg_df = preprocess_sensor_data(resampled_df, config['features']['ppg'],ppg_lowcut,ppg_highcut,freq_ppg)
            temp_df = preprocess_sensor_data(temp_resampled_df, config['features']['temperature'])
            
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
            
            temp_gen = create_sensor_generator(temp_df, config['features']['temperature'], config['labels']['label_column'], 
                                               window_size_temp, step_size_temp)

            # Store each generator output for further processing
            windowed_data.append((acc_gen, gyro_gen, ppg_gen, temp_gen))
            #windowed_data.append((acc_gen, gyro_gen, ppg_gen))
            
            total_steps = total_steps + len(acc_df) // (window_size_acc * config['windowing']['batch_size'])
            
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
    
    output_types = ((tf.float32, tf.float32, tf.float32,tf.float32), tf.int64)
    output_shapes = (
        (
            (window_size_acc, len(config['features']['accelerometer']), 1), 
            (window_size_gyro, len(config['features']['gyroscope']), 1), 
            (window_size_ppg, len(config['features']['ppg']), 1), 
            (window_size_temp, len(config['features']['temperature']), 1)
        ), 
        ()
    )
    
    for (acc_gen, gyro_gen, ppg_gen, temp_gen) in windowed_data:
        combined_dataset = create_combined_tf_dataset([acc_gen, gyro_gen, ppg_gen, temp_gen], output_types, output_shapes)
        datasets.append(combined_dataset)
        
    #for (acc_gen, gyro_gen, ppg_gen) in windowed_data:
    #    combined_dataset = create_combined_tf_dataset([acc_gen, gyro_gen, ppg_gen], output_types, output_shapes)
    #    datasets.append(combined_dataset)
    
    if datasets:
        final_dataset = datasets[0]
        for ds in datasets[1:]:
            final_dataset = final_dataset.concatenate(ds)
        return final_dataset
    else:
        return None

import pandas as pd
import numpy as np

def synchronize_sensors(df, config, resample_freq='100L'):
    """
    Synchronize sensor data by resampling to a common time grid and interpolating.
    :param df: DataFrame containing sensor data and Unix timestamps.
    :param config: Configuration dictionary with sensor features.
    :param resample_freq: The frequency for resampling (e.g., '100L' for 100 ms).
    :return: Synchronized DataFrame.
    """
    # Convert Unix times to datetime
    df['datetime'] = pd.to_datetime(df['unixTimes'], unit='ms')
    
    # Define the common time grid
    common_time_grid = pd.date_range(start=df['datetime'].min(), end=df['datetime'].max(), freq=resample_freq)
    
    # Initialize an empty DataFrame for synchronized data with the common time grid
    synchronized_data = pd.DataFrame({'datetime': common_time_grid})

    # Interpolate and align sensor data on the common time grid
    for sensor_group in ['accelerometer', 'gyroscope', 'ppg']:
        sensor_columns = config['features'][sensor_group]
        
        # For each sensor, resample and interpolate based on 'datetime'
        for sensor in sensor_columns:
            sensor_data = df[['datetime', sensor]].dropna()
            sensor_data_resampled = pd.merge_asof(synchronized_data[['datetime']], sensor_data, on='datetime', direction='nearest')
            sensor_data_resampled[sensor] = sensor_data_resampled[sensor].interpolate(method='linear')
            synchronized_data[sensor] = sensor_data_resampled[sensor]
    
    # Align and interpolate 'sleep_label' column
    df['sleep_label'] = df['sleep_label'].ffill().bfill()  # Forward fill and backward fill
    sleep_label_resampled = pd.merge_asof(synchronized_data[['datetime']], df[['datetime', 'sleep_label']], on='datetime', direction='nearest')
    synchronized_data['sleep_label'] = sleep_label_resampled['sleep_label']

    # Convert back to Unix times
    synchronized_data['unixTimes'] = synchronized_data['datetime'].astype(np.int64) // 10**6

    return synchronized_data

# Example usage
# df = your_dataframe_with_data
# config = load_config()
# synchronized_data = synchronize_sensors(df, config, resample_freq='100L')


    
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
    directory_path = './data/train'
    
    windowed_data, total_steps, input_shapes = process_each_file(directory_path, config)
    
    print("total_steps",total_steps)
    print("input_shapes",input_shapes)
    # Create combined datasets for all files
    final_dataset = process_and_create_datasets(windowed_data, config)
    
    
    
    for data in final_dataset.take(1):
        #print(data)
        features, label = data
        for i, feature_set in enumerate(features):
            print(f"Features shape for sensor {i+1}:", feature_set.numpy().shape)
        print("Label:", label.numpy())
        break

    all_labels = []
    for _, label in final_dataset:
        all_labels.append(label.numpy())
    all_labels = np.array(all_labels)
    
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
    class_weight_dict = dict(enumerate(class_weights))
    print("Class Weights:", class_weight_dict)









    
    
    
    
