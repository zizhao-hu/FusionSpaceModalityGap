import pandas as pd
import os

# Path to the CSV file
csv_path = "vis/t2i/occupation/results/classification_results_raw.csv"

# Check if the file exists
if not os.path.exists(csv_path):
    print(f"Error: CSV file not found at {csv_path}")
    exit(1)

# Create a backup of the original file
backup_path = csv_path + ".backup"
try:
    with open(csv_path, 'r') as src, open(backup_path, 'w') as dst:
        dst.write(src.read())
    print(f"Created backup at {backup_path}")
except Exception as e:
    print(f"Error creating backup: {str(e)}")
    exit(1)

# Read the CSV file
try:
    df = pd.read_csv(csv_path)
    print(f"Read {len(df)} rows from {csv_path}")
except Exception as e:
    print(f"Error reading CSV file: {str(e)}")
    exit(1)

# Check if the image_path column exists
if 'image_path' not in df.columns:
    print("Error: 'image_path' column not found in the CSV file")
    exit(1)

# Count how many paths need to be updated
old_path_count = df['image_path'].str.contains('data/coco/occupation').sum()
print(f"Found {old_path_count} paths that need to be updated")

# Update the paths
df['image_path'] = df['image_path'].str.replace('data/coco/occupation', 'vis/t2i/occupation', regex=False)

# Save the updated CSV file
try:
    df.to_csv(csv_path, index=False)
    print(f"Updated CSV file saved to {csv_path}")
except Exception as e:
    print(f"Error saving updated CSV file: {str(e)}")
    exit(1)

print("Path update completed successfully!") 