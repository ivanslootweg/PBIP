import os
import shutil
from pathlib import Path

# Define paths
# Mapping based on user context: /data/pathology/projects/ivan/WSS/PBIP -> Z:/ivan/WSS/PBIP
base_dir = Path('Z:/ivan/WSS/PBIP/runs')

# Defining the names based on the error log provided by the user
bad_uid = "500_th0500_th0None-109950_top_attention_74c0dda5-109950_top_attention_74c0dda5"
correct_uid = "500_th0None-109950_top_attention_74c0dda5"

bad_dir_path = base_dir / bad_uid
correct_dir_path = base_dir / correct_uid

def rename_artifacts():
    if not base_dir.exists():
        print(f"Error: Base directory not found at {base_dir}")
        return

    # 1. Rename the directory
    if bad_dir_path.exists():
        if correct_dir_path.exists():
            print(f"Warning: Target directory {correct_dir_path} already exists.")
            # Move content from bad to correct? Or just assume correct is better? 
            # User said "execute the renaming", implying the bad folder holds the data they want.
            # I will rename the bad folder to a temp name, then move its content.
            # Actually, safer to just stop if target exists to avoid overwriting good data.
            print("Please check manually if you want to merge.")
            return
        
        print(f"Renaming directory:\n  {bad_dir_path}\n  -> {correct_dir_path}")
        bad_dir_path.rename(correct_dir_path)
    elif correct_dir_path.exists():
        print(f"Directory already seems to be correct: {correct_dir_path}")
    else:
        print(f"Error: Could not find bad directory: {bad_dir_path}")
        # Try to find by partial match?
        pass

    # 2. Rename files inside the correct directory
    if correct_dir_path.exists():
        image_features_dir = correct_dir_path / "image_features"
        if image_features_dir.exists():
            for file_path in image_features_dir.glob("*"):
                if bad_uid in file_path.name:
                    new_name = file_path.name.replace(bad_uid, correct_uid)
                    new_file_path = file_path.with_name(new_name)
                    print(f"Renaming file:\n  {file_path.name}\n  -> {new_name}")
                    file_path.rename(new_file_path)
        else:
            print(f"Warning: {image_features_dir} does not exist.")
            
        # Check if there are other places needing rename?
        # prototypes dir?
        prototypes_dir = correct_dir_path / "prototypes"
        if prototypes_dir.exists():
             for file_path in prototypes_dir.glob("*"):
                if bad_uid in file_path.name:
                    new_name = file_path.name.replace(bad_uid, correct_uid)
                    new_file_path = file_path.with_name(new_name)
                    print(f"Renaming file:\n  {file_path.name}\n  -> {new_name}")
                    file_path.rename(new_file_path)

if __name__ == "__main__":
    rename_artifacts()
