import os
import json
import numpy as np
from collections import defaultdict

def find_optimal_threshold(data_folder):
    # Dictionary to store differences for each threshold
    threshold_differences = defaultdict(lambda: {"TST": [], "WASO": []})

    # Iterate through all JSON files in the folder
    for file_name in os.listdir(data_folder):
        if file_name.endswith(".json"):
            # Extract the threshold from the file name
            parts = file_name.split("_")
            threshold = float(parts[2].replace("threhsold", "").replace(".json", ""))
            
            # Load the JSON file
            file_path = os.path.join(data_folder, file_name)
            with open(file_path, "r") as f:
                data = json.load(f)
            
            # Iterate through the data and calculate differences
            for participant, metrics in data.items():
                gt_tst = metrics["Ground Truth"]["TST (min)"]
                pred_tst = metrics["Predicted"]["TST (min)"]
                gt_waso = metrics["Ground Truth"]["WASO (min)"]
                pred_waso = metrics["Predicted"]["WASO (min)"]
                
                # Calculate absolute differences
                tst_diff = abs(gt_tst - pred_tst)
                waso_diff = abs(gt_waso - pred_waso)
                
                # Append differences to the threshold
                threshold_differences[threshold]["TST"].append(tst_diff)
                threshold_differences[threshold]["WASO"].append(waso_diff)

    # Calculate mean differences for each threshold
    mean_differences = {}
    for threshold, diffs in threshold_differences.items():
        mean_tst_diff = np.mean(diffs["TST"])
        mean_waso_diff = np.mean(diffs["WASO"])
        mean_differences[threshold] = {"TST": mean_tst_diff, "WASO": mean_waso_diff}

    # Find the threshold with the smallest mean differences
    optimal_threshold_tst = min(mean_differences, key=lambda x: mean_differences[x]["TST"])
    optimal_threshold_waso = min(mean_differences, key=lambda x: mean_differences[x]["WASO"])

    return mean_differences, optimal_threshold_tst, optimal_threshold_waso

if __name__ == "__main__":
    data_folder = "results/SleepMetrices/Thresholds"
    mean_differences, optimal_tst, optimal_waso = find_optimal_threshold(data_folder)

    # Print the results
    print("Mean Differences by Threshold:")
    for threshold, diffs in sorted(mean_differences.items()):
        print(f"Threshold {threshold}: TST Diff = {diffs['TST']:.2f}, WASO Diff = {diffs['WASO']:.2f}")

    print(f"\nOptimal Threshold for TST: {optimal_tst}")
    print(f"Optimal Threshold for WASO: {optimal_waso}")