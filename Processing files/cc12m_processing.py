import os
import tarfile
import json
from tqdm import tqdm  # For progress tracking

# Base directory containing tar files
base_dir = "/scratch/group/optmai/datasets/cc12m_webdataset/cc12m/"

# Output JSON file
output_json = "cc12m_full_data.json"

# Initialize list for all entries
data_entries = []

# Unique ID counter
image_id_counter = 0

# Get list of tar files
tar_files = sorted([f for f in os.listdir(base_dir) if f.endswith(".tar")])

# Iterate through all tar files in the base directory with a progress bar
for tar_file in tqdm(tar_files, desc="Processing .tar files", unit="file"):
    tar_path = os.path.join(base_dir, tar_file)
    
    # Open the tar file without extracting
    with tarfile.open(tar_path, "r") as tar:
        # Get members of the tar file
        members = tar.getmembers()
        
        # Process .txt and .jpg pairs with an inner progress bar
        for member in tqdm(members, desc=f"Processing {tar_file}", unit="file", leave=False):
            if member.name.endswith(".txt"):
                # Read caption
                txt_file = tar.extractfile(member)
                caption = txt_file.read().decode("utf-8").strip()
                
                # Corresponding image file
                image_file_name = member.name.replace(".txt", ".jpg")
                image_path = f"{tar_file}/{image_file_name}"
                
                # Add entry to the list
                data_entries.append({
                    "caption": caption,
                    "image": image_path,
                    "image_id": f"cc12m_{image_id_counter}"
                })
                
                # Increment image ID counter
                image_id_counter += 1

# Save all entries to JSON file
with open(output_json, "w") as json_file:
    json.dump(data_entries, json_file, indent=4)

print(f"Finished processing. Saved {len(data_entries)} entries to {output_json}")

