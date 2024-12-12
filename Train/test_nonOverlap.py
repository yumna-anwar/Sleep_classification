import pandas as pd
import numpy as np
import os
import tensorflow as tf
import json
from Data_loader_new import *
from model import *
#from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter 
from sklearn.metrics import f1_score
from collections import defaultdict
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
import argparse


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

def aggregate_predictions(y_pred, window_size, step_size):
    """
    Aggregate predictions over non-overlapping 5-second intervals.
    
    Args:
    y_pred (list or np.array): Overlapping predictions, one per window (e.g., 60-second windows).
    window_size (int): Size of the window in seconds (e.g., 60s).
    step_size (int): Step size for the sliding window in seconds (e.g., 5s).
    
    Returns:
    np.array: Aggregated predictions for non-overlapping 5-second intervals.
    """
    num_expansions = window_size // step_size  # Number of 5-second intervals in a 60-second window
    signal_length = len(y_pred) * step_size  # Total signal length in 5-second increments

    aggregated_pred = np.zeros(signal_length)

    for idx, pred in enumerate(y_pred):
        # Expand and aggregate each prediction for the corresponding 5-second intervals
        start_idx = idx * step_size
        end_idx = start_idx + window_size
        aggregated_pred[start_idx:end_idx] += pred
    
    # Normalizing the aggregated predictions
    aggregated_pred /= num_expansions
    
    return aggregated_pred

def convert_aggregated_to_binary(aggregated_scores, threshold=0.5):
    """
    Convert aggregated scores back to binary (0 or 1) based on a majority threshold.

    Args:
    aggregated_scores (np.array): Aggregated scores for non-overlapping windows.
    threshold (float): Threshold to convert to binary, default is 0.5 (i.e., majority rule).

    Returns:
    np.array: Binary values (0 or 1) based on the threshold.
    """
    return (aggregated_scores >= threshold).astype(int)
def plot_ground_truth_vs_predictions(y_true, y_pred, save_path):
    """
    Plot ground truth vs aggregated non-overlapping predictions.

    Args:
    y_true (list or np.array): Ground truth labels.
    y_pred (list or np.array): Aggregated predicted labels.
    save_path (str): Path to save the plot.
    """
    plt.figure(figsize=(12, 6))

    windows = np.arange(len(y_true))

    plt.plot(windows, y_true, label="Ground Truth", color='blue', marker='o', linestyle='-', markersize=2)
    plt.plot(windows, y_pred, label="Predicted", color='orange', marker='x', linestyle='--', markersize=2)

    plt.xlabel('5-second Window Number (non-overlapping)')
    plt.ylabel('Labels (Awake=0, Sleep=1)')
    plt.title('Ground Truth vs Predictions (Non-Overlapping Windows)')
    plt.legend()

    plt.savefig(save_path)
    plt.close()

def plot_ground_truth(axes, time_indices, ground_truth, sleep_classes):
    """Plot binary ground truth hypnogram with color-coded sleep classes."""
    for i in range(len(time_indices)):
        cls = sleep_classes[i]  # Get the sleep class for the segment
        color = sleep_class_colors[cls]  # Get the color for the sleep class
        axes.plot(
            [time_indices[i], time_indices[i] + 1],  # Plot segment
            [ground_truth[i], ground_truth[i]],  # Binary values (0 or 1)
            color=color,
            linewidth=2,
        )
    axes.set_ylabel("Ground Truth (0=Awake, 1=Sleep)")
    axes.grid(True)
    custom_lines = [Line2D([0], [0], color=color, lw=2) for cls, color in sleep_class_colors.items()]
    axes.legend(custom_lines, sleep_class_colors.keys(), loc="upper right", title="Sleep Stages")

def calculate_tst_and_waso(labels, sampling_rate):
    """
    Calculate Total Sleep Time (TST) and Wake After Sleep Onset (WASO) from sleep labels.

    Args:
    labels (list): List of sleep labels (0 for awake, 1 for sleep).
    sampling_rate (int): Number of label values per second (e.g., 25 for 25 Hz).

    Returns:
    tuple: TST (minutes), WASO (minutes)
    """
    # Convert sampling rate to a multiplier for seconds
    samples_per_minute = sampling_rate * 60

    # Count total sleep samples
    total_sleep_samples = sum(1 for label in labels if label == 1)

    # Calculate WASO
    wake_after_sleep_onset_samples = 0
    sleep_started = False

    for label in labels:
        if label == 1:  # Sleep
            if not sleep_started:
                sleep_started = True  # Mark the start of sleep
        elif label == 0 and sleep_started:  # Wake after sleep onset
            wake_after_sleep_onset_samples += 1

    # Convert samples to minutes
    total_sleep_time_minutes = total_sleep_samples / samples_per_minute
    wake_after_sleep_onset_minutes = wake_after_sleep_onset_samples / samples_per_minute

    return total_sleep_time_minutes, wake_after_sleep_onset_minutes

def calculate_sleep_metrics(labels, sampling_rate):
    """
    Calculate sleep metrics based on ground truth or predicted labels.

    Args:
    labels (list): List of sleep labels (0 for awake, 1 for sleep).
    sampling_rate (int): Number of label values per second.

    Returns:
    dict: A dictionary containing calculated sleep metrics.
    """
    # Convert sampling rate to time units
    time_per_sample = 1 / sampling_rate  # Seconds per sample
    time_per_minute = 60  # Seconds per minute

    # TIB (Time in Bed) in minutes
    tib = len(labels) * time_per_sample / time_per_minute

    # Identify sleep epochs
    sleep_indices = [i for i, label in enumerate(labels) if label == 1]
    if not sleep_indices:
        # If no sleep epochs, return zeros for sleep-related metrics
        return {
            'TST (min)': 0,
            'WASO (min)': 0,
            'TIB (min)': tib,
            'SE (%)': 0,
            'SO (min)': 0,
            'Awakenings': 0
        }

    # Sleep onset time (SO)
    first_sleep_index = sleep_indices[0]
    so = first_sleep_index * time_per_sample / time_per_minute

    # Total Sleep Time (TST) in minutes
    tst = len(sleep_indices) * time_per_sample / time_per_minute

    # WASO (Wake After Sleep Onset)
    wake_indices_after_onset = [
        i for i in range(first_sleep_index, len(labels)) if labels[i] == 0
    ]
    waso = len(wake_indices_after_onset) * time_per_sample / time_per_minute

    # Sleep Efficiency (SE) as a percentage
    se = (tst / tib) * 100 if tib > 0 else 0

    # Awakenings: Transitions from sleep to wake after sleep onset
    awakenings = 0
    in_sleep = True
    for i in range(first_sleep_index, len(labels)):
        if labels[i] == 0 and in_sleep:
            awakenings += 1
            in_sleep = False
        elif labels[i] == 1:
            in_sleep = True

    return {
        'TST (min)': tst,
        'WASO (min)': waso,
        'TIB (min)': tib,
        'SE (%)': se,
        'SO (min)': so,
        'Awakenings': awakenings
    }

def plot_bland_altman(ground_truth_values, predicted_values, metric_name):
    """
    Plot a Bland-Altman plot for comparing two sets of measurements.

    Args:
    ground_truth_values (list): List of ground truth values.
    predicted_values (list): List of predicted values.
    metric_name (str): Name of the metric being compared (e.g., "TST" or "WASO").
    """
    mean_values = (np.array(ground_truth_values) + np.array(predicted_values)) / 2
    differences = np.array(ground_truth_values) - np.array(predicted_values)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)

    # Bland-Altman plot
    plt.figure(figsize=(8, 6))
    plt.scatter(mean_values, differences, alpha=0.6, label="Differences")
    plt.axhline(mean_diff, color='red', linestyle='--', label=f"Mean Diff = {mean_diff:.2f}")
    plt.axhline(mean_diff + 1.96 * std_diff, color='green', linestyle='--', label=f"+1.96 SD = {mean_diff + 1.96 * std_diff:.2f}")
    plt.axhline(mean_diff - 1.96 * std_diff, color='blue', linestyle='--', label=f"-1.96 SD = {mean_diff - 1.96 * std_diff:.2f}")
    plt.title(f"Bland-Altman Plot for {metric_name}")
    plt.xlabel(f"Mean {metric_name} (minutes)")
    plt.ylabel(f"Difference in {metric_name} (Ground Truth - Predicted)")
    plt.legend()
    plt.grid(True)
    #plt.show()
    output_filename = f"results/hypnograms/{metric_name}_TST_WASO.png"
    plt.savefig(output_filename)
    print(f"Saved TST_WASO to {output_filename}")

def calculate_average_awake_time(labels, sampling_rate):
    """
    Calculate the total awake time for a single participant in minutes.

    Args:
    labels (list): List of labels for a single participant (0 for awake, 1 for sleep).
    sampling_rate (int): Number of label values per second (e.g., 25 for 25 Hz).

    Returns:
    float: Total awake time in minutes.
    """

    samples_per_minute = sampling_rate * 60

    awake_samples = sum(1 for label in labels if label == 0)
    awake_time_minutes = awake_samples / samples_per_minute
    return awake_time_minutes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run sleep detection analysis.")
    parser.add_argument('--fold_num', type=int, default=5, help='Fold number to use for evaluation.')
    parser.add_argument('--prediction_threshold', type=float, default=0.55, help='Prediction threshold for classification.')
    args = parser.parse_args()
    fold_num = args.fold_num
    prediction_threshold = args.prediction_threshold
    
      # Load configuration
    config = load_config()
    random_seed = 100
    features_acc = config['features']['accelerometer']
    features_gyro = config['features']['gyroscope']
    features_ppg = config['features']['ppg']
    features_temp = config['features']['temperature']
    label_col = config['labels']['label_column']
    config['windowing']['window_size_seconds'] = 60
    config['windowing']['step_size_seconds'] = 60#config['windowing']['window_size_seconds']//4
    batch_size = 64#config['batch_size']
    class_names = ['Awake','Sleep' ]  # Replace with your actual class names

    config['segmentation']['segment_length_seconds'] = 30
    config['segmentation']['step_size_seconds'] = config['segmentation']['segment_length_seconds']//4
    config['synthetic_data']['duration'] = config['segmentation']['segment_length_seconds']
    config['scaling']=False

    if fold_num==1:
        test_directory_paths = ['./data/5folds/fold5/']
        mod_name = 'best_model_smallNew2_removedNS_highpass01_order5_NoPeakRemoval_ppghighpass02low5_RobScaleAll_BinFocalLossG4a02_lrdecayCos12ka05_win60_step30_batch32_FilterSegment30_lr0.0001'
    elif fold_num==2:
        test_directory_paths = ['./data/5folds/fold1/']
        mod_name = 'fold2_best_model_smallNew2_removedNS_highpass01_order5_NoPeakRemoval_ppghighpass02low5_RobScaleAll_BinFocalLossG4a02_lrdecayCos12ka05_win60_step30_batch32_FilterSegment30_lr0.0001'
    elif fold_num==3:
        test_directory_paths = ['./data/5folds/fold2/']
        mod_name = 'fold3_best_model_smallNew2_removedNS_highpass01_order5_NoPeakRemoval_ppghighpass02low5_RobScaleAll_BinFocalLossG4a02_lrdecayCos12ka05_win60_step30_batch32_FilterSegment30_lr0.0001'
    elif fold_num==4:
        test_directory_paths = ['./data/5folds/fold3/']
        mod_name = 'fold4_best_model_smallNew2_removedNS_highpass01_order5_NoPeakRemoval_ppghighpass02low5_RobScaleAll_BinFocalLossG4a02_lrdecayCos12ka05_win60_step30_batch32_FilterSegment30_lr0.0001'
    elif fold_num==5:
        test_directory_paths = ['./data/5folds/fold4/']
        mod_name = 'fold5_best_model_smallNew2_removedNS_highpass01_order5_NoPeakRemoval_ppghighpass02low5_RobScaleAll_BinFocalLossG4a02_lrdecayCos12ka05_win60_step30_batch32_FilterSegment30_lr0.0001'

    model = tf.keras.models.load_model('models/folds/'+mod_name+'.h5')
    model.summary()

    sleep_class_colors = {
    'WK': 'blue',
    'REM': 'green',
    'NS': 'pink',
    'N1': 'purple',
    'N2': 'red',
    'N3': 'orange',
    }

#     sleep_class_colors = {
#         'Awake': 'blue',
#         'Sleep': 'green',
#         'Deep Sleep': 'orange',
#     }

    simplified_classes = {
        'WK': 'Awake',
        'NS': 'Sleep',
        'REM': 'Sleep',
        'N1': 'Sleep',
        'N2': 'Sleep',
        'N3': 'Deep Sleep',
    }

    fname = 'fold5_smallNew_noScaling_overlap'
    plot_bool = False
    plot_fp_bool = False
    # Main computation loop
    ground_truth_tst = []
    predicted_tst = []
    ground_truth_waso = []
    predicted_waso = []
    results = {}
    for test_directory in test_directory_paths:
        for test_file in os.listdir(test_directory):
            if test_file.endswith('.csv'):
                file_path = os.path.join(test_directory, test_file)
                print(file_path)
                
                # Dictionary to hold false positive sleep class counts
                false_positive_classes = defaultdict(int)
                total_sleep_class_counts = defaultdict(int)
                time_indices = []
                ground_truth_windows = []
                predicted_labels_expanded = []
                predicted_probabilities = []
                actual_labels_expanded = []
                sleep_classes_expanded = []
                for (features, label, ground_truth_window,sleep_class_window, (start_idx, end_idx)) in process_per_file(file_path, config):

                    acc_window, gyro_window, ppg_window, temp_window = features

                    # Prepare inputs by expanding dimensions to simulate a batch of size 1
                    acc_window = np.expand_dims(acc_window, axis=0)  # Add batch dimension
                    gyro_window = np.expand_dims(gyro_window, axis=0)
                    ppg_window = np.expand_dims(ppg_window, axis=0)
                    temp_window = np.expand_dims(temp_window, axis=0)

                    # Model input as a list of the expanded dimensions
                    model_input = [acc_window, gyro_window, ppg_window, temp_window]
                    
                    # Predict using the model
                    pred_prob = model.predict(model_input,verbose=0)
                    # Predict using the model

                    pred_label = 1 if pred_prob[0] >= prediction_threshold else 0

                    # Expand predicted and actual labels to the window size
                    window_time = list(range(start_idx, end_idx))
                    expanded_pred_label = [pred_label] * len(window_time)
                    expanded_actual_label = [label] * len(window_time)
                    expanded_pred_prob = [pred_prob[0]] * len(window_time) 
                    
                    # Append data to lists for plotting
                    time_indices.extend(window_time)
                    ground_truth_windows.extend(ground_truth_window)
                    predicted_labels_expanded.extend(expanded_pred_label)
                    predicted_probabilities.extend(expanded_pred_prob)
                    actual_labels_expanded.extend(expanded_actual_label)
                    sleep_classes_expanded.extend(sleep_class_window)

                    # Check for false positives and categorize by sleep class
                    # Update total sleep class counts
                    for cls in sleep_class_window.values:
                        total_sleep_class_counts[cls] += 1
                    for i in range(len(expanded_pred_label)):
                        if expanded_pred_label[i] == 0 and expanded_actual_label[i] == 1:
                            false_positive_classes[sleep_class_window.values[i]] += 1

                    #if end_idx>50000:
                    #    break
                tst_gt, waso_gt = calculate_tst_and_waso(ground_truth_windows,sampling_rate=25)
                tst_pred, waso_pred = calculate_tst_and_waso(predicted_labels_expanded,sampling_rate=25)
                # Append to lists for Bland-Altman analysis
                ground_truth_tst.append(tst_gt)
                predicted_tst.append(tst_pred)
                ground_truth_waso.append(waso_gt)
                predicted_waso.append(waso_pred)
                gt_metrics = calculate_sleep_metrics(ground_truth_windows, sampling_rate=25)
                pred_metrics = calculate_sleep_metrics(predicted_labels_expanded, sampling_rate=25)

                # Store results for this participant
                results[test_file] = {
                    'Ground Truth': gt_metrics,
                    'Predicted': pred_metrics
                }

                print(f"Participant: {test_file}")
                print(f"  TST (Ground Truth): {tst_gt:.2f} min, TST (Predicted): {tst_pred:.2f} min")
                print(f"  WASO (Ground Truth): {waso_gt:.2f} min, WASO (Predicted): {waso_pred:.2f} min")
                # Calculate average awake time
                avg_awake_actual = calculate_average_awake_time(ground_truth_windows, sampling_rate=25)
                avg_awake_actual_win = calculate_average_awake_time(actual_labels_expanded, sampling_rate=25)
                avg_awake_predicted = calculate_average_awake_time(predicted_labels_expanded, sampling_rate=25)

                print(f"Average Awake Time (Actual): {avg_awake_actual:.2f} minutes")
                print(f"Average Awake Time (Actual after window): {avg_awake_actual_win:.2f} minutes")
                print(f"Average Awake Time (Predicted): {avg_awake_predicted:.2f} minutes")
    
                if plot_fp_bool:
                # Data for plotting
                    sleep_classes = list(false_positive_classes.keys())
                    raw_counts = list(false_positive_classes.values())
                    total_instances = Counter(sleep_classes_expanded)

                    normalized_false_positive_classes = {
                        cls: false_positive_classes[cls] / total_instances[cls]
                        for cls in false_positive_classes.keys()
                    }

                    # Prepare data
                    sleep_classes = list(false_positive_classes.keys())
                    false_positive_counts = [false_positive_classes[cls] for cls in sleep_classes]
                    normalized_counts = [normalized_false_positive_classes[cls] for cls in sleep_classes]

                    # Bar colors based on sleep class colors
                    colors = [sleep_class_colors[cls] for cls in sleep_classes]

                    # Create the figure and axis
                    fig, ax1 = plt.subplots(figsize=(12, 8))

                    # Plot normalized false positives on the primary y-axis
                    bars = ax1.bar(sleep_classes, normalized_counts, color=colors, alpha=0.7, label="Normalized False Positives")
                    ax1.set_ylabel("Normalized False Positives", color='blue')
                    ax1.tick_params(axis='y', labelcolor='blue')

                    # Add secondary y-axis for actual counts
                    ax2 = ax1.twinx()
                    ax2.plot(
                        sleep_classes,
                        false_positive_counts,
                        color='black',
                        marker='o',
                        linestyle='--',
                        linewidth=2,
                        label="Actual False Positive Counts"
                    )
                    ax2.set_ylabel("Actual False Positive Counts", color='black')
                    ax2.tick_params(axis='y', labelcolor='black')

                    # Add total instances as a text box within the plot
                    total_instances_text = "\n".join([f"{cls}: {total_instances[cls]}" for cls in sleep_classes])
                    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
                    plt.text(-0.5, max(normalized_counts) * 0.9,  # Adjust position dynamically
                             f"Total Instances:\n{total_instances_text}", fontsize=10, bbox=props)

                    # Add legend
                    legend_elements = [Patch(facecolor=color, label=cls) for cls, color in zip(sleep_classes, colors)]
                    ax2.legend(
                        handles=legend_elements + [plt.Line2D([0], [0], color='black', linestyle='--', 
                                                              label='Actual False Positive Counts')],
                        loc="upper right",
                        title="Legend"
                    )

                    # Add title and grid
                    plt.title("Normalized and Actual False Positive Distribution by Sleep Class")
                    plt.xlabel("Sleep Classes")
                    ax1.grid(axis='y')

                    # Save the plot
                    plt.savefig(f"results/hypnograms/{test_file}_false_positive_distribution_dual_axis.png")

                    plt.show()
                    #break
                    
                if plot_bool:
                    ground_truth_count = Counter(ground_truth_windows)
                    predicted_count = Counter(predicted_labels_expanded)
                    actual_count = Counter(actual_labels_expanded)
                    print(f"Ground Truth Counts: {ground_truth_count}")
                    print(f"Predicted Counts: {predicted_count}")
                    print(f"Actual Counts: {actual_count}")

                    # Create the stacked Hypnogram plot
                    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
                    fig.suptitle(f"Hypnogram for {test_file}", fontsize=16)

                    # Plot Ground Truth
    #                 axes[0].plot(time_indices, ground_truth_windows, label="Ground Truth", color="blue")
    #                 axes[0].set_ylabel("Ground Truth")
    #                 axes[0].legend(loc="upper right")
    #                 axes[0].grid(True)

                    # Ground Truth
                    #plot_ground_truth(axes[0], time_indices, ground_truth_windows, sleep_classes_expanded)

                     # Plot Ground Truth with Color-Coded Sleep Classes
                    for i in range(len(time_indices)):
                        axes[0].plot(
                            [time_indices[i], time_indices[i] + 1],  # Plot as a step to separate segments
                            [ground_truth_windows[i], ground_truth_windows[i]],
                            color=sleep_class_colors[sleep_classes_expanded[i]],
                        )
                    axes[0].set_ylabel("Ground Truth")
                    axes[0].grid(True)

                    # Add legend for sleep classes
                    custom_lines = [Line2D([0], [0], color=color, lw=4) for color in sleep_class_colors.values()]
                    axes[0].legend(custom_lines, sleep_class_colors.keys(), loc="upper right", title="Sleep Classes")


                    # Plot Predicted Labels (Expanded)
                    axes[1].plot(time_indices, predicted_labels_expanded, label="Predicted Labels", color="orange", linestyle="--")
                    axes[1].plot(time_indices, predicted_probabilities, label="Predicted Probabilities", color="red", alpha=0.6)
                    axes[1].set_ylabel("Predicted")
                    axes[1].legend(loc="upper right")
                    axes[1].grid(True)

                    # Plot Actual Labels (Expanded)
                    axes[2].plot(time_indices, actual_labels_expanded, label="Actual Labels", color="green")
                    axes[2].set_ylabel("Actual")
                    axes[2].legend(loc="upper right")
                    axes[2].grid(True)

                    # Set shared x-axis label
                    axes[2].set_xlabel("Time (indices)")

                    # Save the plot
                    output_filename = f"results/hypnograms/{test_file}_noNS_hypnogram.png"
                    plt.savefig(output_filename)
                    print(f"Saved hypnogram to {output_filename}")
                    plt.close()
                

    # Filepath to save the JSON file
    output_file = f"results/SleepMetrices/results_sleepMetrices_threhsold{prediction_threshold}_fold{fold_num}.json"

    # Write JSON to the file
    with open(output_file, "w") as file:
        json.dump(results, file, indent=4)  # indent=4 makes the JSON pretty-printed
    # Bland-Altman plots
    plot_bland_altman(ground_truth_tst, predicted_tst, "TST")
    plot_bland_altman(ground_truth_waso, predicted_waso, "WASO")
    
    
