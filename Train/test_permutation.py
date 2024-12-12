import pandas as pd
import numpy as np
import os
import tensorflow as tf
import json
from Data_loader import *
from model import *
#from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter 
from sklearn.metrics import f1_score


def plot_confusion_matrix(conf_matrix, labels,fname):
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    plt.savefig("results/"+fname+"_Conf_mat.png")
    plt.close()
    
def plot_roc_curve(fpr, tpr, roc_auc, filename):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig(filename)
    plt.close()

def calculate_sensitivity_specificity_at_thresholds(fpr, tpr, thresholds, specificities):
    sensitivities = {}
    for spec in specificities:
        # Find the threshold where specificity is closest to the desired level
        idx = np.where(fpr <= 1 - spec)[0]
        if len(idx) == 0:
            sensitivities[spec] = 0.0
        else:
            sensitivity = tpr[idx[-1]]
            sensitivities[spec] = sensitivity
            print(f"At {spec*100}% specificity, threshold: {thresholds[idx[-1]]}, sensitivity: {sensitivity}")
    return sensitivities

def _generate_random_sensor_data(sensor_data, data_range=None):
    """
    Replace the given sensor data with random noise that mimics the original data distribution.
    Noise is generated with mean 0 and scaled to fit within the specified range.
    
    Args:
    sensor_data (numpy array): The original sensor data to be replaced with random noise.
    data_range (tuple or None): Optional range for scaling the noise. If None, default scaling will be used.
    
    Returns:
    numpy array: Randomized sensor data with noise that mimics the original data.
    """
    # Calculate the mean and standard deviation of the original data
    mean = np.mean(sensor_data, axis=0)
    std_dev = np.std(sensor_data, axis=0)
    
    # Generate noise with mean 0
    noise = np.random.normal(loc=0, scale=1, size=sensor_data.shape)
    
    # Scale noise to fit within the batch-specific 95% range of the original data
    scale_factor = (std_dev * 2)  # Assuming 95% falls within ±2 std deviations

    # Apply scaling and shift by the original mean
    random_sensor_data = mean + noise * scale_factor
    
    return random_sensor_data

def generate_random_sensor_data(sensor_data):
    """
    Replace the given sensor data with random noise, scaled within the min-max range of the batch.
    
    Args:
    sensor_data (numpy array): The original sensor data to be replaced with random noise.
    
    Returns:
    numpy array: Randomized sensor data with values scaled to the min-max range.
    """
    # Calculate min and max for the batch
    min_val = np.min(sensor_data, axis=0)
    max_val = np.max(sensor_data, axis=0)
    
    # Generate random values between 0 and 1
    noise = np.random.uniform(low=0, high=1, size=sensor_data.shape)
    
    # Scale the random noise to fit within the min-max range
    random_sensor_data = min_val + noise * (max_val - min_val)
    
    return random_sensor_data

def _plot_comprehensive_bar(original_scores, sensor_scores, fname):
    """
    Create a bar plot comparing accuracy and F1-scores for original and sensor-masked experiments.
    """
    labels = ['Original', 'Accelerometer Masked', 'Gyroscope Masked', 'ppg Masked', 'Temperature Masked']
    accuracy_vals = [original_scores['accuracy']] + [sensor_scores[i]['accuracy'] for i in range(4)]
    f1_awake_vals = [original_scores['f1_awake']] + [sensor_scores[i]['f1_awake'] for i in range(4)]
    f1_sleep_vals = [original_scores['f1_sleep']] + [sensor_scores[i]['f1_sleep'] for i in range(4)]
    
    x = np.arange(len(labels))  # Label locations
    width = 0.2  # Width of the bars
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Bar plots
    ax.bar(x - width, accuracy_vals, width, label='Accuracy')
    ax.bar(x, f1_awake_vals, width, label='F1 Awake')
    ax.bar(x + width, f1_sleep_vals, width, label='F1 Sleep')

    # Add labels, title, and legend
    ax.set_xlabel('Model Variants')
    ax.set_ylabel('Scores')
    ax.set_title('Comparison of Accuracy and F1-Scores for Masked Sensors')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim([0.5, 1.0])
    plt.savefig(f'results/{fname}_comprehensive_comparison.png')
    plt.close()
    
def plot_comprehensive_bar(original_scores, sensor_scores, fname):
    """
    Create a bar plot comparing accuracy and F1-scores for original and sensor-masked experiments.
    """
    labels = ['Original', 'Accelerometer Masked', 'Gyroscope Masked', 'PPG Masked', 'Temperature Masked']
    
    # Extracting accuracy and F1 scores
    accuracy_vals = [original_scores['accuracy']] + [sensor_scores[i]['accuracy'] for i in range(4)]
    f1_awake_vals = [original_scores['f1_awake']] + [sensor_scores[i]['f1_awake'] for i in range(4)]
    f1_sleep_vals = [original_scores['f1_sleep']] + [sensor_scores[i]['f1_sleep'] for i in range(4)]

    # Calculate differences with the original scores
    accuracy_diffs = [0] + [sensor_scores[i]['accuracy'] - original_scores['accuracy'] for i in range(4)]
    f1_awake_diffs = [0] + [sensor_scores[i]['f1_awake'] - original_scores['f1_awake'] for i in range(4)]
    f1_sleep_diffs = [0] + [sensor_scores[i]['f1_sleep'] - original_scores['f1_sleep'] for i in range(4)]
    
    x = np.arange(len(labels))  # Label locations
    width = 0.2  # Width of the bars
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Bar plots
    accuracy_bars = ax.bar(x - width, accuracy_vals, width, label='Accuracy')
    f1_awake_bars = ax.bar(x, f1_awake_vals, width, label='F1 Awake')
    f1_sleep_bars = ax.bar(x + width, f1_sleep_vals, width, label='F1 Sleep')

    # Add labels, title, and legend
    ax.set_xlabel('Model Variants')
    ax.set_ylabel('Scores')
    ax.set_title('Comparison of Accuracy and F1-Scores for Masked Sensors')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim([0.4, 1.0])  # Set y-axis starting point to 0.4 for clarity

    # Annotate bars with the differences
    for i in range(len(labels)):
        # Accuracy difference annotation
        ax.annotate(f"{accuracy_diffs[i]:.2f}",
                    xy=(accuracy_bars[i].get_x() + accuracy_bars[i].get_width() / 2, accuracy_bars[i].get_height()),
                    xytext=(0, 3),  # Offset to move text slightly above the bar
                    textcoords="offset points",
                    ha='center', va='bottom')
        
        # F1 awake difference annotation
        ax.annotate(f"{f1_awake_diffs[i]:.2f}",
                    xy=(f1_awake_bars[i].get_x() + f1_awake_bars[i].get_width() / 2, f1_awake_bars[i].get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

        # F1 sleep difference annotation
        ax.annotate(f"{f1_sleep_diffs[i]:.2f}",
                    xy=(f1_sleep_bars[i].get_x() + f1_sleep_bars[i].get_width() / 2, f1_sleep_bars[i].get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.savefig(f'results/{fname}_comprehensive_comparison.png')
    plt.close()
    
def evaluate_model_on_dataset(model, test_dataset, sensor_idx_to_randomize=None,min_max_values=None):
    y_true = []
    y_pred = []
    y_pred_prob = []

    for batch in test_dataset:
        x, y = batch
        
        # Randomize the selected sensor's data if sensor_idx_to_randomize is provided
        if sensor_idx_to_randomize is not None and min_max_values is not None:
            sensor_data = x[sensor_idx_to_randomize].numpy()
            
            # Retrieve min and max values for the selected sensor
            min_value = min_max_values[f'sensor_{sensor_idx_to_randomize + 1}']['min']
            max_value = min_max_values[f'sensor_{sensor_idx_to_randomize + 1}']['max']
         
            
            #random_sensor_data = generate_random_sensor_data(sensor_data,(min_value,max_value))
            random_sensor_data = generate_random_sensor_data(sensor_data)
            x = list(x)
            x[sensor_idx_to_randomize] = random_sensor_data

        preds = model.predict(x)
        y_true.extend(y.numpy())
        y_pred_prob.extend(preds)
    
    # Calculate metrics
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    y_pred = (np.array(y_pred_prob) >= optimal_threshold).astype(int)
    
    f1 = f1_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, target_names=['Awake', 'Sleep'], output_dict=True)
    
    return {
        'accuracy': class_report['accuracy'],
        'f1_awake': class_report['Awake']['f1-score'],
        'f1_sleep': class_report['Sleep']['f1-score'],
        'conf_matrix': conf_matrix,
        'roc_auc': roc_auc
    }

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
    config['windowing']['window_size_seconds'] = 60
    config['windowing']['step_size_seconds'] = 5#config['windowing']['window_size_seconds']//4
    batch_size = 64#config['batch_size']
    class_names = ['Awake','Sleep' ]  # Replace with your actual class names

    config['segmentation']['segment_length_seconds'] = 30
    config['segmentation']['step_size_seconds'] = config['segmentation']['segment_length_seconds']//4
    config['synthetic_data']['duration'] = config['segmentation']['segment_length_seconds']
    
    # Directory paths for training, validation, and testing CSV files
    test_directory_path = './data/test/'
    
    test_windowed_data, steps_per_epoch_train,input_shapes = process_each_file(test_directory_path, config)
    test_dataset = process_and_create_datasets(test_windowed_data, config)
    min_max_values = calculate_min_max_values(test_dataset)
    print(min_max_values)
    print(steps_per_epoch_train)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    
    model = model = tf.keras.models.load_model('models/optuna_fixed/best_model_smallNew_NewPPGfilter_win60_batch32_FilterSegment30_zero_weight2_lr0.0001.h5')
    #model = model = tf.keras.models.load_model('models/optuna_fixed/best_model_smallNew_NewPPGfilter_transformer_win60_batch32_FilterSegment30_zero_weight2_lr0.0001.h5')
    #model = model = tf.keras.models.load_model('models/optuna_fixed/best_model_New2_NewPPGfilter_win60_batch32_FilterSegment30_zero_weight2_lr0.0001.h5')
    #model = model = tf.keras.models.load_model('models/optuna_fixed/best_model_smallNew_noTemp_NewPPGfilter_win60_batch32_FilterSegment30_zero_weight2_lr0.0001.h5')
    #model = model = tf.keras.models.load_model('models/optuna_fixed/best_model_smallNew_robustScale_NewPPGfilter_win60_batch32_FilterSegment30_zero_weight2_lr0.0001.h5')
    
    #model = model = tf.keras.models.load_model('models/optuna_fixed/best_model_smallNew_TempDense8_NewPPGfilter_win60_batch32_FilterSegment30_zero_weight2_lr0.0001.h5')
    
    
    # Evaluate original model without randomizing any sensors
    original_scores = evaluate_model_on_dataset(model, test_dataset)
    print(original_scores)
    
    # Evaluate models by randomizing sensors (0 to 3)
    sensor_scores = {}
    for sensor_idx in range(4):
        sensor_scores[sensor_idx] = evaluate_model_on_dataset(model, test_dataset, 
                                                              sensor_idx_to_randomize=sensor_idx,
                                                              min_max_values=min_max_values)
        
        # Save individual sensor results
        fname = f'sensor_{sensor_idx}_masked'
        print(sensor_scores)
        #plot_comprehensive_bar(original_scores, sensor_scores, fname=fname)

    # Generate final comprehensive comparison plot
    plot_comprehensive_bar(original_scores, sensor_scores, fname='final_comparison')
        
