import os
import glob

def clean_folder(folder_path, keep_count=100):
    """Clean a folder by keeping only the first N images."""
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return
    
    # Get all image files
    image_files = sorted([f for f in os.listdir(folder_path) 
                         if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    if len(image_files) <= keep_count:
        print(f"{folder_path}: Already has {len(image_files)} images (≤ {keep_count})")
        return
    
    # Get files to delete (everything after the first keep_count)
    files_to_delete = image_files[keep_count:]
    
    print(f"{folder_path}: Removing {len(files_to_delete)} files...")
    for file_name in files_to_delete:
        file_path = os.path.join(folder_path, file_name)
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {str(e)}")
    
    print(f"{folder_path}: Cleanup complete. {keep_count} images remaining.")

def main():
    # Base directory containing all the image folders
    base_dir = "data/coco"
    
    # Folders to clean
    folders = [
        # CFG variations
        "generated_sd1_4_1",
        "generated_sd1_4_3",
        "generated_sd1_4_7",
        "generated_sd1_4_10",
        "generated_sd1_4_20",
        # Step variations
        "generated_sd1_4_7_steps_10",
        "generated_sd1_4_7_steps_20",
        "generated_sd1_4_7_steps_50",
        "generated_sd1_4_7_steps_100",
        "generated_sd1_4_7_steps_200",
        "generated_sd1_4_7_steps_500"
    ]
    
    print("Starting folder cleanup...")
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        clean_folder(folder_path)
    
    print("\nCleanup completed for all folders.")

if __name__ == "__main__":
    main() 