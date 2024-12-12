import pandas as pd
import numpy as np
import os
import tensorflow as tf
import json
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard
from datetime import datetime
import optuna
from optuna.integration import TFKerasPruningCallback
import joblib
from tensorflow.keras.callbacks import EarlyStopping
from optuna.trial import FixedTrial
from tensorflow.keras.models import load_model
from tensorflow.keras import mixed_precision
import gc
from tensorflow.keras import backend as K
import itertools
import random
import threading
import time
import GPUtil
import psutil
import matplotlib.pyplot as plt

#FROM PROJECT FILES
from Data_loader_new import *
from model import *

#mixed_precision.set_global_policy('mixed_float16')
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/ebuild/software/CUDA/11.7.0"
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.random.set_seed(1)

class GarbageCollectionCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        tf.keras.backend.clear_session()
        gc.collect()
        
# Initialize monitoring data
monitoring_data = {"time_points": [], "gpu_usage": [], "gpu_memory": [], "ram_usage": []}
monitoring_flag = True  # Toggle monitoring on/off


monitoring_lock = threading.Lock()

def monitor_resources(interval=1):
    global monitoring_flag
    while monitoring_flag:
        with monitoring_lock:
            # Update monitoring_data
            gpus = GPUtil.getGPUs()
            monitoring_data["time_points"].append(time.time())
            if gpus:
                gpu = gpus[0]
                monitoring_data["gpu_usage"].append(gpu.memoryUsed / 1024)  # Convert MB to GB
                monitoring_data["gpu_memory"].append(gpu.memoryTotal / 1024)  # Total GPU memory in GB
            else:
                monitoring_data["gpu_usage"].append(0)
                monitoring_data["gpu_memory"].append(0)
            monitoring_data["ram_usage"].append(psutil.virtual_memory().percent)
            time.sleep(interval)
    print("Monitoring thread stopped.")



def plot_and_save_monitoring():
    time_points = np.array(monitoring_data["time_points"]) - monitoring_data["time_points"][0]

    # Align lengths of all metrics
    min_length = min(len(time_points), len(monitoring_data["gpu_usage"]), len(monitoring_data["gpu_memory"]), len(monitoring_data["ram_usage"]))
    time_points = time_points[:min_length]
    monitoring_data["gpu_usage"] = monitoring_data["gpu_usage"][:min_length]
    monitoring_data["gpu_memory"] = monitoring_data["gpu_memory"][:min_length]
    monitoring_data["ram_usage"] = monitoring_data["ram_usage"][:min_length]

    plt.figure(figsize=(12, 6))

    # GPU Memory Usage
    plt.subplot(3, 1, 1)
    plt.plot(time_points, monitoring_data["gpu_usage"], label="GPU Memory Usage (GB)", color="blue")
    plt.ylabel("GPU Usage (GB)")
    plt.legend()

    # Total GPU Memory
    plt.subplot(3, 1, 2)
    plt.plot(time_points, monitoring_data["gpu_memory"], label="Total GPU Memory (GB)", color="green")
    plt.ylabel("Total GPU Memory (GB)")
    plt.legend()

    # RAM Usage
    plt.subplot(3, 1, 3)
    plt.plot(time_points, monitoring_data["ram_usage"], label="RAM Usage (%)", color="red")
    plt.ylabel("RAM Usage (%)")
    plt.xlabel("Time (s)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("resource_monitoring_plots.png")
    plt.savefig("resource_monitoring_plots.pdf")
    plt.close()
    print("Resource monitoring plots saved.")


        
def load_or_create_cache(dataset_fn, cache_path):
    if os.path.exists(cache_path):
        print(f"Loading cached dataset from {cache_path}")
        dataset = tf.data.experimental.load(cache_path)
    else:
        print(f"Creating new dataset and caching to {cache_path}")
        dataset = dataset_fn()
        tf.data.experimental.save(dataset, cache_path)
    return dataset

def process_class_samples(directory_paths, config, class_label, windows_per_shuffle=200):
    """Generator to process windows across multiple files and yield samples with shuffling."""
    windows = []
    
    # Iterate over all files in directory paths
    for data, label in process_each_file(directory_paths, config):
        if label == class_label:
            windows.append((data, label))
            
        # Shuffle and yield windows once we have a buffer of `windows_per_shuffle`
        if len(windows) >= windows_per_shuffle:
            random.shuffle(windows)  # Shuffle the collected windows
            for window in windows:
                yield window
            windows = []  # Reset window buffer after yielding
    
    # Yield any remaining windows after finishing all files
    if windows:
        random.shuffle(windows)
        for window in windows:
            yield window

def balanced_batch_generator(train_directory_paths, config, batch_size, windows_per_shuffle=50):
    """Generator for class-balanced batches with shuffled windows across files."""
    # Create generators for each class
    sleep_generator = process_class_samples(train_directory_paths, config, class_label=0, windows_per_shuffle=windows_per_shuffle)
    awake_generator = process_class_samples(train_directory_paths, config, class_label=1, windows_per_shuffle=windows_per_shuffle)

    while True:
        # Collect balanced samples from each class
        sleep_batch = list(itertools.islice(sleep_generator, batch_size // 2))
        awake_batch = list(itertools.islice(awake_generator, batch_size // 2))

        # Verify that we have enough samples for a full batch
        if len(sleep_batch) == batch_size // 2 and len(awake_batch) == batch_size // 2:
            batch = sleep_batch + awake_batch
            np.random.shuffle(batch)  # Shuffle within the batch

            # Separate out each sensor window and labels
            acc_data_batch = [x[0][0] for x in batch]  # Extract acc windows
            gyro_data_batch = [x[0][1] for x in batch]  # Extract gyro windows
            ppg_data_batch = [x[0][2] for x in batch]  # Extract ppg windows
            temp_data_batch = [x[0][3] for x in batch]  # Extract temp windows
            labels_batch = [x[1] for x in batch]  # Extract labels

            # Yield in the correct format expected by model.fit
            yield (
                [
                    np.array(acc_data_batch), 
                    np.array(gyro_data_batch), 
                    np.array(ppg_data_batch), 
                    np.array(temp_data_batch)
                ],
                np.array(labels_batch)
            )
        else:
            # Restart generators if one of them is exhausted
            sleep_generator = process_class_samples(train_directory_paths, config, class_label=0, windows_per_shuffle=windows_per_shuffle)
            awake_generator = process_class_samples(train_directory_paths, config, class_label=1, windows_per_shuffle=windows_per_shuffle)

@tf.autograph.experimental.do_not_convert
def objective(trial, config):
    random_seed = 2
    features_acc = config['features']['accelerometer']
    features_gyro = config['features']['gyroscope']
    features_ppg = config['features']['ppg']
    features_temp = config['features']['temperature']
    
    # TUNE WITH OPTUNA
    config['windowing']['window_size_seconds'] = trial.suggest_int('window_size_seconds', 30, 240, step=30)
    config['windowing']['step_size_seconds'] = 30
    
    config['windowing']['batch_size'] = trial.suggest_int('batch_size', 32, 64, step=32)
    
    learning_rate = trial.suggest_categorical('learning_rate', [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001])
    zero_weight = trial.suggest_int('zero_weight', 1, 10, step=1)
    
    
    batch_size = config['windowing']['batch_size']
    window_size_seconds = config['windowing']['window_size_seconds']
    step_size_seconds = config['windowing']['step_size_seconds']
    segment_length_seconds = config['segmentation']['segment_length_seconds']
    
    window_size_acc = int(window_size_seconds * config['frequencies']['accelerometer'])
    step_size_acc = int(step_size_seconds * config['frequencies']['accelerometer'])

    window_size_gyro = int(window_size_seconds * config['frequencies']['gyroscope'])
    step_size_gyro = int(step_size_seconds * config['frequencies']['gyroscope'])

    window_size_ppg = int(window_size_seconds * config['frequencies']['ppg'])
    step_size_ppg = int(step_size_seconds * config['frequencies']['ppg'])

    window_size_temp = int(window_size_seconds * config['frequencies']['temperature'])
    step_size_temp = int(step_size_seconds * config['frequencies']['temperature'])
    
    ppg_input_shape = (window_size_ppg, len(config['features']['ppg']), 1)
    gyro_input_shape = (window_size_gyro, len(config['features']['gyroscope']), 1)
    acc_input_shape = (window_size_acc, len(config['features']['accelerometer']), 1)
    temp_input_shape = (window_size_temp, len(config['features']['temperature']), 1)
    
    input_shapes = [ppg_input_shape, gyro_input_shape, acc_input_shape,temp_input_shape]
    
    
    # Directory paths for training, validation, and testing CSV files
    train_directory_path = ['../data/5folds/fold2/','../data/5folds/fold1/','../data/5folds/fold5/']    
    val_directory_path = ['../data/5folds/fold3/']


    train_dataset = tf.data.Dataset.from_generator(
        lambda: process_each_file(train_directory_path, config),
        output_types=((tf.float32, tf.float32, tf.float32, tf.float32), tf.int64),
        output_shapes=(((None, None, 1), (None, None, 1), (None, None, 1), (None, None, 1)), ())
    )
#     train_dataset = balanced_batch_generator(train_directory_path, config, batch_size)

    
    val_dataset = tf.data.Dataset.from_generator(
        lambda: process_each_file(val_directory_path, config),
        output_types=((tf.float32, tf.float32, tf.float32, tf.float32), tf.int64),
        output_shapes=(((None, None, 1), (None, None, 1), (None, None, 1), (None, None, 1)), ())
    )
      
    train_dataset = train_dataset.cache().shuffle(800).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.cache().shuffle(800).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    
        
    model_config = {
            'lr': learning_rate,
            'num_filters_1': 8,
            'num_filters_2': 16,
            'num_filters_3':32,
            'num_filters_4':64,
            'num_filters_1_sm': 8,
            'num_filters_2_sm': 16,
            'kernel_size_1': (3, 1),
            'kernel_size_2': (3, 1),
            'kernel_size_1_sm': (2, 1),
            'lstm_units': 32,
            'dropout_rate': 0.2,
            'dense_units': 64,
            'l2_regularization': 0.001,
            'pooling_size': (2, 1),
            'num_heads': 8,    # Number of heads in Transformer block
            'ff_dim': 128,     # Feed-forward layer size in Transformer block
            'loss_gamma': 4,
            'loss_alpha': 0.2,
            'lr_alpha': 0.5,
             'lr_decay_steps': 12100
        }
    
    #with strategy.scope():
    model_file_path = f'../models/folds/fold5_best_model_smallNew2_removedNS_highpass01_order5_NoPeakRemoval_ppghighpass02low5_RobScaleAll_BinFocalLossG4a02_lrdecayCos12ka05_win{window_size_seconds}_step{step_size_seconds}_batch{batch_size}_FilterSegment{segment_length_seconds}_lr{learning_rate}.h5'
    
    #model_file_path = f'models/folds/test.h5'

    if os.path.exists(model_file_path):
        print(f"Loading model from {model_file_path}")
        model = load_model(model_file_path)
    else:
        model = build_combined_model(input_shapes, 2,model_config)

    model.summary()

    #class_weight_dict = {0: zero_weight, 1: 1}
    #class_weight_dict = {0: 3.75, 1: 0.57}

    checkpoint = ModelCheckpoint(model_file_path, monitor='val_auc', save_best_only=True, mode='max', verbose=1)


    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    #pruning_callback = TFKerasPruningCallback(trial, 'val_loss', n_warmup_steps=10)

    early_stopping = EarlyStopping(monitor='val_loss', 
                                   patience=10, 
                                   restore_best_weights=True,  
                                   verbose=1)
    garbage_collection_callback = GarbageCollectionCallback()

    history = model.fit(
        train_dataset,
        epochs=500,  
        validation_data=val_dataset,
        callbacks=[checkpoint, early_stopping, garbage_collection_callback],
        #class_weight=class_weight_dict,
        verbose=1      
    )
    
    
    val_loss = history.history['val_loss'][-1]
        
    tf.keras.backend.clear_session()
    
 
    return val_loss

  
if __name__ == '__main__':
    run_study=False
    config = load_config()
    
    if run_study:
        study_name = "Window_batch_lr_smallModel_WithTemp_optimization_study"

        # Create the Optuna study
        study = optuna.create_study(direction='minimize')

        # Optimize the objective function with the config passed in
        study.optimize(lambda trial: objective(trial, config), n_trials=20)

        print('Best trial:')
        trial = study.best_trial

        print(f'  Value: {trial.value}')
        print(f'  Params: ')
        for key, value in trial.params.items():
            print(f'    {key}: {value}')


        joblib.dump(study, "optuna_studies/"+study_name+".pkl")

    else:
        # Fixed parameters
        fixed_params = {
            'window_size_seconds': 60,
            'segment_length_seconds': 30,
            'batch_size':32,
            'learning_rate':0.0001,
            'zero_weight':2
        }

        # Call the objective function manually
        fixed_result = objective(FixedTrial(fixed_params),config)
        print(f'Objective function result with fixed parameters: {fixed_result}')