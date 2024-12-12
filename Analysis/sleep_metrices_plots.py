import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Directory containing the JSON files
data_directory = "results/SleepMetrices"  # Replace with your directory path

# Metrics to analyze
metrics = ["TST (min)", "WASO (min)", "TIB (min)", "SE (%)", "SO (min)", "Awakenings"]
ground_truth = {metric: [] for metric in metrics}
predicted = {metric: [] for metric in metrics}
subject_ids = []  # To track subject IDs

# Read all JSON files in the directory
for file_name in os.listdir(data_directory):
    if file_name.endswith(".json"):  # Process only JSON files
        file_path = os.path.join(data_directory, file_name)
        with open(file_path, "r") as file:
            data = json.load(file)
        print(data)
        # Extract data for each file
        for test_file, values in data.items():
            subject_ids.append(test_file)
            for metric in metrics:
                ground_truth[metric].append(values["Ground Truth"][metric])
                predicted[metric].append(values["Predicted"][metric])

def plot_bland_altman_with_outliers(ground_truth, predicted, metric_name):
    ground_truth = np.array(ground_truth)
    predicted = np.array(predicted)
    mean = (ground_truth + predicted) / 2
    diff = ground_truth - predicted  # Difference between Ground Truth and Predicted
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)

    # Calculate limits
    upper_limit = mean_diff + 1.96 * std_diff
    lower_limit = mean_diff - 1.96 * std_diff

    # Identify outliers
    outliers = [(subject_ids[i], diff[i]) for i in range(len(diff)) if diff[i] > upper_limit or diff[i] < lower_limit]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(mean, diff, alpha=0.6, label="Data points")
    
    # Annotate each point with the corresponding subject ID
    for i, subject_id in enumerate(subject_ids):
        plt.text(mean[i], diff[i], subject_id, fontsize=8, alpha=0.7)
        
    plt.axhline(mean_diff, color='red', linestyle='--', label=f"Mean Diff ({mean_diff:.2f})")
    plt.axhline(upper_limit, color='blue', linestyle='--', label=f"Upper Limit (+1.96 SD)")
    plt.axhline(lower_limit, color='blue', linestyle='--', label=f"Lower Limit (-1.96 SD)")
    
    plt.title(f"Bland-Altman Plot for {metric_name}")
    plt.xlabel("Mean of Ground Truth and Predicted")
    plt.ylabel("Difference (Ground Truth - Predicted)")
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plot_file_path = os.path.join(data_directory, f"Bland_Altman_{metric_name.replace(' ', '_')}.png")
    plt.savefig(plot_file_path)
    plt.close()  # Close the plot to free memory
    print(f"Saved plot for {metric_name} to {plot_file_path}")

    # Print outliers
    if outliers:
        print(f"\nOutliers for {metric_name}:")
        for subject_id, difference in outliers:
            print(f"  Subject ID: {subject_id}, Difference: {difference:.2f}")
    else:
        print(f"\nNo outliers for {metric_name}.")

# Generate Bland-Altman plots for each metric
for metric in metrics:
    plot_bland_altman_with_outliers(ground_truth[metric], predicted[metric], metric)