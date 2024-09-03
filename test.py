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


def plot_confusion_matrix(conf_matrix, labels):
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    plt.savefig("Conf_mat.png")
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


if __name__ == '__main__':
      # Load configuration
    config = load_config()
    random_seed = 100
    features_acc = config['features']['accelerometer']
    features_gyro = config['features']['gyroscope']
    features_ppg = config['features']['ppg']
    features_temp = config['features']['temperature']
    label_col = config['labels']['label_column']
    config['windowing']['window_size_seconds'] = 240#config['windowing']['window_size_seconds']
    config['windowing']['step_size_seconds'] = config['windowing']['window_size_seconds']//2
    batch_size = 64#config['batch_size']
    class_names = ['Awake','Sleep' ]  # Replace with your actual class names

    config['segmentation']['segment_length_seconds'] = 100
    config['segmentation']['step_size_seconds'] = config['segmentation']['segment_length_seconds']//4
    config['synthetic_data']['duration'] = config['segmentation']['segment_length_seconds']
    
    # Directory paths for training, validation, and testing CSV files
    test_directory_path = './data/test/'
    
    test_windowed_data, steps_per_epoch_train,input_shapes = process_each_file(test_directory_path, config)
    test_dataset = process_and_create_datasets(test_windowed_data, config)
    
    print(steps_per_epoch_train)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    # Load the model
    #model = tf.keras.models.load_model('models/best_model_avg_08_step10.h5')
    #model = tf.keras.models.load_model('models/best_model_avg_08_step10_noPPG_BIG.h5')
    #model = tf.keras.models.load_model('models/best_model_avg_08_step10_smallMod_BIG.h5')
    #model = tf.keras.models.load_model('models/best_model_avg_08_step10_smallMod_weight2-1_BIG.h5')
    #model = tf.keras.models.load_model('models/best_model_avg_08_step10_smallMod_weight2-1_lstm_BIG.h5')
    #model = tf.keras.models.load_model('models/best_model_avg_08_step10_california.h5')
    
    #model = tf.keras.models.load_model('models/best_model_bpass_filteredPPG.h5')
    #model = tf.keras.models.load_model('models/best_model_bpass_filteredPPG_gyro2_10.h5')
    #model = tf.keras.models.load_model('models/best_model_bpass_filteredPPG_win60_inc20_batch64.h5')
    #model = tf.keras.models.load_model('models/optuna2/best_model_win240_step120_batch32.h5')
    #model = tf.keras.models.load_model('models/optuna/best_model_win180_step90_batch32.h5')
    
    #model = tf.keras.models.load_model('models/optuna3/best_model_win210_step105_batch32_FilterSegment20_zero_weight1_lr0.0001.h5')
    #model = tf.keras.models.load_model('models/optuna4/best_model_lr0.000011_filters1_32_filters2_16_kernel1_5x1_kernel2_7x1_lstm_128_dropout0.26_dense_128_l2_0.000009_pooling_3x1.h5')
    model = tf.keras.models.load_model('models/optuna6/best_model_win240_batch64_FilterSegment100_zero_weight2.5_lr0.1.h5')
    
    
    
    # Evaluate the model
    y_true = []
    y_pred = []
    y_pred_prob = []

    for batch in test_dataset:
        
        x, y = batch
        preds = model.predict(x)
        y_true.extend(y.numpy())
        # Apply the custom threshold
        y_pred_prob.extend(preds)
        
    label_counts = Counter(y_true)
    print(label_counts)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Plot and save the ROC curve
    plot_roc_curve(fpr, tpr, roc_auc, filename='roc_curve.png')
    
    specificities = [0.6,0.70, 0.80, 0.85, 0.90]
    sensitivities = calculate_sensitivity_specificity_at_thresholds(fpr, tpr, thresholds, specificities)

    for spec, sens in sensitivities.items():
        print(f"Sensitivity at {int(spec * 100)}% Specificity: {sens}")

    # Find the optimal threshold (you can use different strategies here, e.g., Youden's J statistic)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal Threshold: {optimal_threshold}")

    # Apply the optimal threshold to get the final predictions
    y_pred = (np.array(y_pred_prob) >= optimal_threshold).astype(int)
    
    # Calculate confusion matrix and classification report
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, target_names=class_names)

    tn, fp, fn, tp = conf_matrix.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    print("sensitivity",sensitivity)
    print("specificity",specificity)

    print("Confusion Matrix:")
    plot_confusion_matrix(conf_matrix, labels=class_names)
    print("\nClassification Report:")
    print(class_report)


    #Specificity: is the fraction of awake labels correctly identified as awale out of all the actual awake labels.
    #Sensitivity: The proportion of correctly predicted sleep instances (1) out of all actual sleep instances (1).




