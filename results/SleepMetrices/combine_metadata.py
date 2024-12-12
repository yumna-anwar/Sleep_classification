import json
import os

# Paths to directories
data_directory = "Thresholds/optimal/"  # Replace with your folder path
metadata_directory = "extracted_metadata_files/"  # Replace with your folder path
output_directory = "combined_metrices/"  # Replace with your folder path
os.makedirs(output_directory, exist_ok=True)

# Read Ground Truth and Predicted data
combined_data = {}
for file_name in os.listdir(data_directory):
    if file_name.endswith(".json"):
        with open(os.path.join(data_directory, file_name), "r") as file:
            data = json.load(file)
            combined_data.update(data)

meta_fnames=[]
for file_name in os.listdir(metadata_directory):
    if file_name.endswith(".json"):
        meta_fnames.append(file_name.split(".")[0])
print(meta_fnames)
# Add metadata to each participant
for participant, values in combined_data.items():
    print(participant)
    # Extract subject ID (assuming it's part of the filename)
    for meta_fn in meta_fnames:
        if meta_fn.lower() in participant.lower():
            print("found meta data")
            metadata_file = f"{meta_fn}.json"  # Map to the corresponding metadata file
            metadata_path = os.path.join(metadata_directory, metadata_file)
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as meta_file:
                    metadata = json.load(meta_file)
                    metrics = metadata.get("metrics", {})  # Extract metrics
                    combined_data[participant]["Metadata"] = metrics
            else:
                combined_data[participant]["Metadata"] = None  # If no metadata found

# Save the combined data to a new JSON file
output_file = os.path.join(output_directory, "combined_data.json")
with open(output_file, "w") as out_file:
    json.dump(combined_data, out_file, indent=4)

print(f"Combined data saved to {output_file}")