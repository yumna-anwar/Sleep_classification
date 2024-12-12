import pandas as pd
import numpy as np
import os
import tensorflow as tf
import json
from Data_loader_features import *
from model import *
#from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter 
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import joblib

def extract_features_and_labels(tf_dataset):
    """
    Convert a TensorFlow dataset into NumPy arrays for Scikit-learn.
    """
    features, labels = [], []
    for batch_features, batch_labels in tf_dataset:
        features.append(batch_features.numpy())
        labels.append(batch_labels.numpy())
    features = np.vstack(features)  # Stack all batches into a single array
    labels = np.hstack(labels)  # Flatten labels into a single array
    return features, labels

def plot_confusion_matrix(conf_matrix, labels,fname):
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    plt.savefig("results_features/fold5_"+fname+"_Conf_mat.png")
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

def plot_ground_truth_vs_predictions_windowed(y_true, y_pred, save_path):
    """
    Plot ground truth vs predictions based on 60-second windows.

    Args:
    y_true (list or np.array): Ground truth labels.
    y_pred (list or np.array): Predicted labels.
    save_path (str): Path to save the plot.
    """
    plt.figure(figsize=(12, 6))
    
    # Create a list of window numbers
    windows = np.arange(len(y_true))

    # Plot Ground Truth
    plt.plot(windows, y_true, label="Ground Truth", color='blue', marker='o', linestyle='-', markersize=2)

    # Plot Predictions
    plt.plot(windows, y_pred, label="Predicted", color='orange', marker='x', linestyle='--', markersize=2)

    plt.xlabel('Window Number (60s each)')
    plt.ylabel('Labels (Awake=0, Sleep=1)')
    plt.title('Ground Truth vs Predictions')
    plt.legend()

    # Save the plot
    plt.savefig(save_path)
    plt.close()



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
    config['windowing']['step_size_seconds'] = 30#config['windowing']['window_size_seconds']//4
    batch_size = 64#config['batch_size']
    class_names = ['Awake','Sleep' ]  # Replace with your actual class names

    config['segmentation']['segment_length_seconds'] = 30
    config['segmentation']['step_size_seconds'] = config['segmentation']['segment_length_seconds']//4
    config['synthetic_data']['duration'] = config['segmentation']['segment_length_seconds']

    # Directory paths for training, validation, and testing CSV files
    test_directory_path = ['./data/5folds/fold5/']
    
    
    test_dataset = load_all_files_as_dataset(test_directory_path, config, batch_size=32, shuffle=False,buffer_size=10)

    # Extract features and labels
    test_features, test_labels = extract_features_and_labels(test_dataset)

    # Load the saved SVM model
    #mod_name = "svm_model_balanced_noStanScale_Robscale_win60_batch32_C1.00_gammascale"
    #mod_name = "svm_model_balanced_noStanScale_Robscale_win60_batch32_C0.10_gammascale"
    mod_name = "svm_model_balanced_noStanScale_Robscale_win60_batch32_C0.00_gammascale"
    
    model_file_path = "models_features/"+mod_name+".pkl"
    svm_model = joblib.load(model_file_path)
    
    fname = mod_name
    y_true = test_labels
    label_counts = Counter(y_true)
    # Get probabilities or decision scores
    y_pred_prob = svm_model.decision_function(test_features)  # Or use predict_proba if probability=True

    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Plot and save the ROC curve
    plot_roc_curve(fpr, tpr, roc_auc, filename='results_features/fold5_'+fname+'_roc_curve.png')
    
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
    
    f1 = f1_score(y_true, y_pred)
    print(f"F1 Score: {f1:.4f}")
    
    # Calculate confusion matrix and classification report
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, target_names=class_names)

    tn, fp, fn, tp = conf_matrix.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    print("sensitivity",sensitivity)
    print("specificity",specificity)

    print("Confusion Matrix:")
    plot_confusion_matrix(conf_matrix, labels=class_names,fname=fname)
    print("\nClassification Report:")
    print(class_report)

    with open(f'results_features/fold5_{fname}_classification_report.txt', 'w') as f:
        # Write label counts
        f.write(f"Label Counts: {label_counts}\n\n")

        # Write AUC information
        f.write(f"ROC AUC: {roc_auc:.4f}\n\n")

        # Write sensitivity at different specificities
        f.write("Sensitivity at different specificities:\n")
        for spec, sens in sensitivities.items():
            f.write(f"Sensitivity at {int(spec * 100)}% Specificity: {sens:.4f}\n")

        # Write optimal threshold
        f.write(f"\nOptimal Threshold (Youden's J): {optimal_threshold:.4f}\n\n")

        # Write F1 Score
        f.write(f"F1 Score: {f1:.4f}\n\n")

        # Write classification report
        f.write("Classification Report:\n")
        f.write(class_report + "\n")

        # Write confusion matrix
        f.write(f"Confusion Matrix:\n")
        conf_matrix_str = "\n".join(['\t'.join([str(cell) for cell in row]) for row in conf_matrix])
        f.write(conf_matrix_str + "\n\n")

        # Write sensitivity and specificity
        f.write(f"Sensitivity: {sensitivity:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n")
  
    
    
    


