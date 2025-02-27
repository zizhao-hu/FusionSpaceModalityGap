import os
import re

def rename_folders():
    """Rename sd_to_sd folders to include steps_50 if not already present"""
    base_dir = "data/coco"
    
    # Get all directories that match the pattern sd_to_sd_cfg_X_gen_Y
    pattern = r'sd_to_sd_cfg_\d+_gen_\d+'
    
    for item in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, item)) and re.match(pattern, item):
            old_path = os.path.join(base_dir, item)
            # Insert steps_50 before gen_
            new_name = item.replace('_gen_', '_steps_50_gen_')
            new_path = os.path.join(base_dir, new_name)
            
            if not os.path.exists(new_path):
                print(f"Renaming {old_path} to {new_path}")
                os.rename(old_path, new_path)
            else:
                print(f"Warning: {new_path} already exists, skipping rename of {old_path}")

if __name__ == "__main__":
    rename_folders() 