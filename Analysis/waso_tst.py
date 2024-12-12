import os
import pandas as pd
from datetime import datetime

def calculate_wk_time(folder_path):
    """
    Reads all CSV files in a folder, extracts 'unixTime' and 'sleep_stage',
    and calculates the total time in 'WK' and time spent in 'WK' after sleep onset.

    Args:
    folder_path (str): Path to the folder containing CSV files.

    Returns:
    dict: A dictionary with file names as keys and their WK times as values.
    """
    results = {}
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing file: {file_name}")

            # Read the CSV
            df = pd.read_csv(file_path)

            # Ensure the columns exist
            if 'unixTimes' not in df.columns or 'sleep_stage' not in df.columns:
                print(f"File {file_name} does not have the required columns. Skipping.")
                continue
           # Calculate time differences between consecutive rows (convert to seconds)
            df['time_diff'] = df['unixTimes'].diff().fillna(0) / 1000  # Convert ms to seconds

            # Total WK time
            wk_total_time = df[df['sleep_stage'] == 'WK']['time_diff'].sum()

            # Find sleep onset (first non-WK row)
            sleep_onset_index = df[df['sleep_stage'] != 'WK'].index.min()

            # WK time after sleep onset
            wk_after_onset_time = 0
            if pd.notna(sleep_onset_index):  # Ensure there is a sleep onset
                wk_after_onset_time = df.iloc[sleep_onset_index:][df['sleep_stage'] == 'WK']['time_diff'].sum()

            # Convert times to minutes
            wk_total_minutes = wk_total_time / 60
            wk_after_onset_minutes = wk_after_onset_time / 60

            # Store results for this file
            results[file_name] = {
                'WK Total (min)': wk_total_minutes,
                'WK After Onset (min)': wk_after_onset_minutes
            }

    return results

# Example Usage
folder_path = './data/5folds/test/'
results = calculate_wk_time(folder_path)

# Print results
for file, stats in results.items():
    print(f"File: {file}")
    print(f"  Total WK (min): {stats['WK Total (min)']}")
    print(f"  WK After Sleep Onset (min): {stats['WK After Onset (min)']}")
    print()