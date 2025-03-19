import os
import sys
import numpy as np
import torch
import clip
from PIL import Image
import glob
from tqdm.auto import tqdm
import argparse
import json
import re

# Add the repository root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

def load_captions(base_dir):
    """Load captions from the COCO dataset.
    
    Args:
        base_dir: Base directory containing the COCO dataset
        
    Returns:
        Dictionary mapping image IDs to captions and a list of sorted captions
    """
    # Use validation set captions instead of training set
    annotation_file = os.path.join(base_dir, "annotations", "captions_val2014.json")
    if not os.path.exists(annotation_file):
        print(f"Annotation file {annotation_file} not found. Using generic captions.")
        return {}, []
    
    try:
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        
        # Create image_id to caption mapping
        image_id_to_caption = {}
        for ann in annotations['annotations']:
            image_id = ann['image_id']
            caption = ann['caption']
            if image_id not in image_id_to_caption:
                image_id_to_caption[image_id] = []
            image_id_to_caption[image_id].append(caption)
        
        # Get unique image IDs
        unique_image_ids = list(image_id_to_caption.keys())
        
        # Use random seed 42 to select image IDs
        np.random.seed(42)
        selected_image_ids = np.random.choice(unique_image_ids, size=200, replace=False)
        
        # Create a list of (index, image_id, caption) tuples
        sorted_captions = []
        for i, image_id in enumerate(selected_image_ids):
            caption = image_id_to_caption[image_id][0]  # Use the first caption for each image
            sorted_captions.append((i, image_id, caption))
        
        return image_id_to_caption, sorted_captions
    except Exception as e:
        print(f"Error loading captions: {e}")
        return {}, []

def extract_image_id_from_filename(filename):
    """Extract image ID from filename.
    
    Args:
        filename: Image filename
        
    Returns:
        Image ID or None if not found
    """
    # Try to extract image ID from COCO filename format (COCO_train2014_000000123456.jpg)
    match = re.search(r'COCO_(?:train|val)2014_(\d+)\.jpg', filename)
    if match:
        return int(match.group(1))
    
    # Try to extract image ID from evaluation filename format (COCO_eval_000000123456.jpg)
    match = re.search(r'COCO_eval_0*(\d+)\.jpg', filename)
    if match:
        return int(match.group(1))
    
    return None

def extract_embeddings(image_dir, output_file, clip_model, clip_preprocess, captions_dict=None, sorted_captions=None, batch_size=100, max_images=200, device="cuda"):
    """Extract CLIP embeddings for images and their corresponding captions.
    
    Args:
        image_dir: Directory containing the images
        output_file: Path to save the embeddings
        clip_model: CLIP model
        clip_preprocess: CLIP preprocessing function
        captions_dict: Dictionary mapping image IDs to captions
        sorted_captions: List of sorted (index, image_id, caption) tuples
        batch_size: Number of images to process in each batch
        max_images: Maximum number of images to process
        device: Device to use for processing
        
    Returns:
        bool: True if embeddings were successfully extracted and saved, False otherwise
    """
    # Get image files
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    
    # Limit to max_images
    if len(image_files) > max_images:
        image_files = image_files[:max_images]
    
    if len(image_files) < batch_size:
        print(f"Not enough images in {image_dir}. Found {len(image_files)}, need {batch_size}")
        return False
    
    # Randomly select batch_size images
    np.random.seed(42)  # Use fixed seed for reproducibility
    selected_indices = np.random.choice(len(image_files), size=batch_size, replace=False)
    selected_files = [image_files[i] for i in selected_indices]
    
    # Load images
    images = []
    for img_path in tqdm(selected_files, desc=f"Loading images for embedding extraction"):
        try:
            img = Image.open(img_path).convert('RGB')
            images.append(img)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    
    if not images:
        print(f"No valid images loaded from {image_dir}")
        return False
    
    # Extract image filenames to get corresponding captions
    image_filenames = [os.path.basename(f) for f in selected_files]
    
    # Create captions for the images
    captions = []
    for i, img_file in enumerate(image_filenames):
        # Try to extract image ID from filename
        image_id = extract_image_id_from_filename(img_file)
        
        # Get caption based on the numeric part of the filename
        # For example, COCO_eval_000000000095.jpg should use the caption at index 95
        if sorted_captions and image_id is not None and image_id < len(sorted_captions):
            # Use the caption from the sorted list
            _, original_image_id, caption = sorted_captions[image_id]
        elif captions_dict and image_id and image_id in captions_dict:
            # Fallback to the caption dictionary if available
            caption = captions_dict[image_id][0]
        else:
            # Use a generic caption if we can't find the actual one
            caption = "A photograph"
        
        captions.append(caption)
        
        # Print example captions for the first two images
        if i < 2:
            print(f"Example {i+1}: Image file: {img_file}")
            print(f"Example {i+1}: Image ID: {image_id}")
            print(f"Example {i+1}: Caption: {caption}")
            if sorted_captions and image_id is not None and image_id < len(sorted_captions):
                print(f"Example {i+1}: Original Image ID: {sorted_captions[image_id][1]}")
            print()
    
    # Process in smaller batches
    clip_batch_size = 16
    
    # Store image and text features
    all_image_features = []
    all_text_features = []
    
    try:
        for i in range(0, len(images), clip_batch_size):
            end_idx = min(i + clip_batch_size, len(images))
            clip_batch_images = images[i:end_idx]
            clip_batch_captions = captions[i:end_idx]
            
            # Preprocess images and text
            processed_images = torch.stack([clip_preprocess(img) for img in clip_batch_images]).to(device)
            text_tokens = clip.tokenize(clip_batch_captions).to(device)
            
            # Get embeddings
            with torch.no_grad():
                image_features = clip_model.encode_image(processed_images)
                text_features = clip_model.encode_text(text_tokens)
            
            # Store features
            all_image_features.append(image_features.cpu().numpy())
            all_text_features.append(text_features.cpu().numpy())
        
        # Concatenate features
        all_image_features = np.vstack(all_image_features)
        all_text_features = np.vstack(all_text_features)
        
        # Save embeddings
        np.savez(
            output_file,
            image_embeddings=all_image_features,
            text_embeddings=all_text_features,
            image_files=selected_files,
            captions=captions
        )
        
        print(f"Saved embeddings to {output_file}")
        return True
    
    except Exception as e:
        print(f"Error extracting embeddings: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='Extract CLIP embeddings for generated images')
    parser.add_argument('--base_dir', type=str, default=os.path.join("data", "coco"), 
                        help='Base directory containing the image folders')
    parser.add_argument('--output_dir', type=str, default=os.path.join("data", "embeddings"),
                        help='Output directory for embeddings')
    parser.add_argument('--target_generations', type=str, default="0,1,2,3,4,5,6,7,8,9,10",
                        help='Comma-separated list of generations to extract embeddings for')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Number of images to process in each batch')
    parser.add_argument('--max_images', type=int, default=200,
                        help='Maximum number of images to consider per generation')
    parser.add_argument('--force_extract', action='store_true',
                        help='Force extraction of embeddings even if they already exist')
    parser.add_argument('--debug', action='store_true',
                        help='Print additional debugging information')
    
    args = parser.parse_args()
    
    # Parse target generations
    target_generations = [int(g) for g in args.target_generations.split(',')]
    print(f"Target generations: {target_generations}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define model name
    model_fname = "openai_clip-vit-base-patch32"
    
    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        print("Loading CLIP model...")
        model_name = "ViT-B/32"
        clip_model, clip_preprocess = clip.load(model_name, device=device, download_root=os.path.expanduser("~/.cache/clip"))
        print(f"CLIP model {model_name} loaded successfully")
    except Exception as e:
        print(f"Error loading CLIP model: {str(e)}")
        return
    
    # Load captions
    print("Loading captions...")
    captions_dict, sorted_captions = load_captions(args.base_dir)
    print(f"Loaded {len(captions_dict)} image captions and {len(sorted_captions)} sorted captions")
    
    # Print some example caption keys for debugging
    if args.debug:
        if captions_dict:
            print("Example caption keys:")
            keys = list(captions_dict.keys())[:5]
            for key in keys:
                print(f"  {key}: {captions_dict[key][0]}")
        
        if sorted_captions:
            print("Example sorted captions:")
            for i, (index, image_id, caption) in enumerate(sorted_captions[:5]):
                print(f"  {index}: Image ID {image_id}: {caption}")
    
    # Define the groups and their suffixes
    groups = [
        {"key": "recursive", "suffix": ""},
        {"key": "finetune", "suffix": "_finetune"},
        {"key": "baseline", "suffix": "_baseline"}
    ]
    
    # Path to the large evaluation directory
    gen_large_dir = os.path.join(args.base_dir, "gen_large")
    
    # Extract embeddings for each generation and group
    for gen in target_generations:
        for group in groups:
            group_key = group["key"]
            group_suffix = group["suffix"]
            
            # Determine the input directory name
            input_dir_name = f"sd_to_sd_cfg_7_steps_50_gen_{gen}{group_suffix}"
            input_dir = os.path.join(gen_large_dir, input_dir_name)
            
            # Determine the output file path
            output_file = os.path.join(args.output_dir, f"CLIP_{model_fname}_embeddings_gen_{gen}{group_suffix}.npz")
            
            # Check if embeddings already exist
            if os.path.exists(output_file) and not args.force_extract:
                print(f"Embeddings already exist at {output_file}. Skipping extraction. Use --force_extract to override.")
                continue
            
            # Check if directory exists and has enough images
            if not os.path.exists(input_dir):
                print(f"Directory {input_dir} does not exist. Skipping.")
                continue
            
            image_files = glob.glob(os.path.join(input_dir, "*.jpg"))
            if len(image_files) < args.batch_size:
                print(f"Not enough images in {input_dir}. Found {len(image_files)}, need {args.batch_size}. Skipping.")
                continue
            
            print(f"Extracting embeddings for generation {gen}, group {group_key} from {input_dir}")
            
            # Extract embeddings
            success = extract_embeddings(
                input_dir, output_file, clip_model, clip_preprocess, captions_dict, sorted_captions,
                batch_size=args.batch_size, max_images=args.max_images, device=device
            )
            
            if success:
                print(f"Successfully extracted embeddings for generation {gen}, group {group_key}")
            else:
                print(f"Failed to extract embeddings for generation {gen}, group {group_key}")
    
    print("Embedding extraction complete!")

if __name__ == "__main__":
    main() 