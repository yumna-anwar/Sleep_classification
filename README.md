Config.json
  - contains all the configurable parameters
Data_loader.py
  - load_config(config_file='config.json')
    Loads configuration settings from the specified JSON file.
  
  - read_and_preprocess_files(directory_path)
    Reads and concatenates data from all CSV files in the given directory. Adds a sleep_label column to classify awake (WK) and sleep stages.
  
  - butter_bandpass(lowcut, highcut, fs, order=5) and bandpass_filter(data, lowcut, highcut, fs, order=5)
    Creates and applies a bandpass filter to the data to filter out unwanted frequencies in the signal.
  
  - preprocess_sensor_data(df, features, lowcut=None, highcut=None, fs=25)
    Preprocesses sensor data by applying a bandpass filter (if specified), normalizing, and cleaning the data.
  
  - calculate_frequency(data)
    Calculates the sampling frequency of the given data based on Unix timestamps.
  
  - create_sensor_generator(data, features, label_col, window_size, step_size)
    Generates TensorFlow-compatible sensor data windows and their corresponding labels for model training.
  
  - create_combined_tf_dataset(generators, output_types, output_shapes)
    Combines multiple sensor generators into a single TensorFlow dataset for simultaneous multi-sensor input.
  
  - calculate_label_distribution(df, label_col)
    Calculates and prints label distribution statistics for data imbalances.
  
  - resample_data(df, target_freq, columns_to_resample, time_col='unixTimes')
    Resamples the specified sensor columns to the target frequency, ensuring uniform sampling across all sensors.
  
  - synchronize_sensors(df, config, resample_freq='100L')
    Resamples all sensor data to a common time grid for synchronization across different sensor streams.
  
  - process_each_file(directory_path, config)
    Processes each file in the directory, resamples, and prepares the sensor data for training, including frequency adjustment and windowing.
  
  - process_and_create_datasets(windowed_data, config)
    Combines all processed data into TensorFlow datasets, ready for training.


    
