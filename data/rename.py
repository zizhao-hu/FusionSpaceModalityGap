import os
import json
import shutil
from glob import glob

def create_mixed_dataset(train_dir, gen_dir, scale, num_total=10000, gen_percent=0.1):
    """
    Create a mixed dataset with a specified percentage of generated images.
    The generated images are renamed to match the corresponding train2014 images based on order.
    """
    # Calculate number of images from each source
    num_gen = int(num_total * gen_percent)
    num_original = num_total - num_gen

    # Create new directory for the mixed dataset
    new_dir = f"finetune_10%gen_scale={scale}"
    os.makedirs(new_dir, exist_ok=True)

    # Get a sorted list of all train images (limit to num_total)
    train_total_files = sorted(glob(os.path.join(train_dir, "*.jpg")))[:num_total]

    # Get the generated images (sorted) and limit to the number we need
    gen_files = sorted(glob(os.path.join(gen_dir, "*.jpg")))[:num_gen]
    # The remaining images will be taken from the train directory,
    # starting after the ones whose names will be used for generated images.
    train_files = train_total_files[num_gen:]

    print(f"\nCreating {new_dir}:")
    print(f"Using {len(gen_files)} generated images and {len(train_files)} original images")

    # Copy generated images and rename them using the train image names based on order
    for i, src in enumerate(gen_files):
        # Use the filename from the corresponding train image
        new_name = os.path.basename(train_total_files[i])
        dst = os.path.join(new_dir, new_name)
        shutil.copy2(src, dst)

    # Copy original train images (they already have the correct name)
    for src in train_files:
        filename = os.path.basename(src)
        dst = os.path.join(new_dir, filename)
        shutil.copy2(src, dst)

    return new_dir

def create_filtered_annotations(orig_anno_path, new_dir, num_images=10000):
    """
    Create a new annotation file containing only the specified number of images.
    """
    # Load original annotations
    with open(orig_anno_path, 'r') as f:
        annotations = json.load(f)

    # Get list of image files in new directory
    valid_images = set(os.path.basename(f) for f in glob(os.path.join(new_dir, "*.jpg")))

    # Filter annotations: only keep images (and their captions) present in the new folder
    filtered_annotations = annotations.copy()
    filtered_annotations['images'] = [
        img for img in annotations['images']
        if img['file_name'] in valid_images
    ]
    valid_image_ids = {img['id'] for img in filtered_annotations['images']}
    filtered_annotations['annotations'] = [
        anno for anno in annotations['annotations']
        if anno['image_id'] in valid_image_ids
    ]

    # Save new annotations
    new_anno_path = f"annotations_{os.path.basename(new_dir)}.json"
    with open(new_anno_path, 'w') as f:
        json.dump(filtered_annotations, f)

    print(f"Created annotation file: {new_anno_path}")
    print(f"Contains {len(filtered_annotations['images'])} images and "
          f"{len(filtered_annotations['annotations'])} captions")

    return new_anno_path

def main():
    # Original directories and annotation file
    train_dir = "train2014"
    anno_file = "annotations/captions_train2014.json"

    # Define the scales and their corresponding generated directories
    scales = [1.0, 3.0, 5.0]
    gen_dirs = [f"generated_scale={scale}" for scale in scales]

    # Create datasets for each scale
    for scale, gen_dir in zip(scales, gen_dirs):
        # Create the mixed dataset with renamed generated images
        new_dir = create_mixed_dataset(
            train_dir=train_dir,
            gen_dir=gen_dir,
            scale=scale,
            num_total=10000,
            gen_percent=0.1
        )

        # Create the corresponding annotation file for the new mixed dataset
        create_filtered_annotations(
            orig_anno_path=anno_file,
            new_dir=new_dir,
            num_images=10000
        )

if __name__ == "__main__":
    main()
