import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def save_analysis_and_metrics_for_all(df_list, results_list, subject_ids, output_file):
    """
    Save DataFrame summaries and calculated metrics for all subjects to a single text file.

    Args:
    df_list (list of pd.DataFrame): List of DataFrames, one for each subject.
    results_list (list of dict): List of results dictionaries, one for each subject.
    subject_ids (list of str): List of subject identifiers.
    output_file (str): Path to the output text file.
    """
    with open(output_file, "w") as file:
        for idx, subject_id in enumerate(subject_ids):
            file.write(f"Subject ID: {subject_id}\n\n")
            
            # Write the DataFrame description
            file.write("DataFrame Summary:\n")
            description = df_list[idx].describe().applymap(lambda x: f"{x:.2f}")
            file.write(description.to_string())
            file.write("\n\n")
            
            # Write the results
            file.write("Calculated Metrics:\n")
            for key, value in results_list[idx].items():
                file.write(f"{key}: {value:.2f}\n")
            
            file.write("\n" + "="*40 + "\n\n")  # Add a separator for clarity
    print(f"Results and summaries for all subjects saved to {output_file}")
def calculate_tst_and_awake(df, subject_id):
    """
    Calculate TST, Total Awake Time, and their ratio based on sleep stages and unix time.

    Args:
    df (pd.DataFrame): DataFrame containing 'unixTimes' and 'sleep_stage'.
    subject_id (str): Subject identifier for reporting.

    Returns:
    dict: Calculated metrics for TST, Total Awake Time, and ratio.
    """
    # Ensure the 'unixTimes' column is sorted
    df = df.sort_values(by='unixTimes')
    
    # Calculate time differences in seconds
    df['time_diff'] = df['unixTimes'].diff().fillna(0) / 1000  # Convert from ms to seconds

    # Assign binary labels for sleep stages (e.g., WK=0 for Awake, others=1 for Sleep)
    sleep_labels = {'WK': 0, 'N1': 1, 'N2': 1, 'N3': 1, 'REM': 1}  # Adjust as per your data
    df['binary_stage'] = df['sleep_stage'].map(sleep_labels)
    
    if df['binary_stage'].isnull().any():
        print(f"Warning: Unmapped sleep stages in {subject_id}. Check data for invalid values.")
    
    # Calculate TST and Total Awake Time in minutes
    total_sleep_time = df[df['binary_stage'] == 1]['time_diff'].sum() / 60  # Sleep time in minutes
    total_awake_time = df[df['binary_stage'] == 0]['time_diff'].sum() / 60  # Awake time in minutes

    # Calculate Sleep/Awake Ratio
    if total_awake_time > 0:
        sleep_awake_ratio = total_sleep_time / total_awake_time
    else:
        sleep_awake_ratio = float('inf')  # Handle edge case where no awake time is recorded
    
    # Print and return the results
    print(f"Subject: {subject_id}")
    print(f"  Total Sleep Time (TST): {total_sleep_time:.2f} minutes")
    print(f"  Total Awake Time: {total_awake_time:.2f} minutes")
    print(f"  Sleep/Awake Ratio: {sleep_awake_ratio:.2f}")

    return {
        "TST (min)": total_sleep_time,
        "Total Awake Time (min)": total_awake_time,
        "Sleep/Awake Ratio": sleep_awake_ratio
    }

# Path to the folder containing subject CSV files
data_folder = "data/All_data"

# List of outlier subject IDs
outlier_subject_ids = [
    "california-00000187-right-sync.csv",
    "california-00011978-right-sync.csv",
    "california-00000234-right-sync.csv",
    "california-00000227-right-sync.csv",
    "california-00011955-right-sync.csv"
    # Add more subject IDs here
]

# Columns of interest
columns_of_interest = ["unixTimes", "sleep_stage", "accelerometerX", "accelerometerY", "accelerometerZ",
                        "gyroscopeX", "gyroscopeY", "gyroscopeZ", "ledIR", "ledRed", "ledGreen", "tempObject"]

output_file = "results/all_subjects_analysis.txt"
df_list = []  # Store DataFrames for all subjects
results_list = []  # Store results for all subjects
subject_ids = []  # Store subject IDs
# Load and summarize data for each outlier
for subject_id in outlier_subject_ids:
    file_path = f"{data_folder}/{subject_id}"
    print(f"Analyzing subject: {subject_id}")

    # Load the data
    df = pd.read_csv(file_path, usecols=columns_of_interest)
    
    # Display unique sleep stages
    print(f"Unique sleep stages for {subject_id}: {df['sleep_stage'].unique()}")

    # Check for invalid sleep stage values
    invalid_values = df['sleep_stage'][~df['sleep_stage'].isin(['WK', 'N1', 'N2', 'N3', 'REM', 'NS'])]
    # Count invalid sleep stages
    invalid_count = len(invalid_values)

    if invalid_count > 0:
        print(f"Invalid sleep stage values detected for {subject_id}: {invalid_count} entries")
        print(invalid_values)
    else:
        print(f"All sleep stage values are valid for {subject_id}")
    
    # Drop rows with invalid sleep stages
    valid_sleep_stages = ['WK', 'N1', 'N2', 'N3', 'REM', 'NS']
    df = df[df['sleep_stage'].isin(valid_sleep_stages)]
    
    # Convert unixTimes to datetime for better readability
    df['datetime'] = pd.to_datetime(df['unixTimes'], unit='ms')
    df.set_index('datetime', inplace=True)
    
    temp_df = df[['unixTimes'] + ['tempObject'] + ['sleep_stage']].dropna()
    # Summary statistics
    print(df.describe())

    results = calculate_tst_and_awake(df, subject_id)
    print(results)
    # Append to the lists
    df_list.append(df)
    results_list.append(results)
    subject_ids.append(subject_id)
    
    # Plot sensor data and sleep stages
    fig, axes = plt.subplots(5, 1, figsize=(15, 12), sharex=True)
    
    # Accelerometer
    axes[0].plot(df.index, df['accelerometerX'], label='Acc X')
    axes[0].plot(df.index, df['accelerometerY'], label='Acc Y')
    axes[0].plot(df.index, df['accelerometerZ'], label='Acc Z')
    axes[0].set_title('Accelerometer Data')
    axes[0].legend()
    axes[0].grid()

    # Gyroscope
    axes[1].plot(df.index, df['gyroscopeX'], label='Gyro X')
    axes[1].plot(df.index, df['gyroscopeY'], label='Gyro Y')
    axes[1].plot(df.index, df['gyroscopeZ'], label='Gyro Z')
    axes[1].set_title('Gyroscope Data')
    axes[1].legend()
    axes[1].grid()

    # PPG
    axes[2].plot(df.index, df['ledIR'], label='IR')
    axes[2].plot(df.index, df['ledRed'], label='Red')
    axes[2].plot(df.index, df['ledGreen'], label='Green')
    axes[2].set_title('PPG Data')
    axes[2].legend()
    axes[2].grid()

    # Temperature
    axes[3].plot(temp_df.index, temp_df['tempObject'], label='Temperature', color='orange')
    axes[3].set_title('Temperature Data')
    axes[3].legend()
    axes[3].grid()

    # Sleep stages
    axes[4].plot(df.index, df['sleep_stage'], label='Sleep Stage', color='purple')
    axes[4].set_title('Sleep Stages')
    axes[4].legend()
    axes[4].grid()

    plt.tight_layout()
    plt.savefig(f"results/{subject_id}_analysis.png")
    plt.show()
    
save_analysis_and_metrics_for_all(df_list, results_list, subject_ids, output_file)