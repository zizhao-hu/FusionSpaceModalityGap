import os
import glob
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import platform
import re
from tqdm import tqdm
import clip
from sklearn.preprocessing import normalize
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "clipscore"))
from clipscore import get_clip_score, extract_all_images, get_refonlyclipscore

def load_coco_captions(annotation_file="data/coco/annotations/captions_train2014.json"):
    """Load COCO captions from json file"""
    print(f"Loading captions from {annotation_file}")
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
        
        print(f"Loaded {len(image_id_to_caption)} image captions")
        return image_id_to_caption
    except Exception as e:
        print(f"Error loading COCO captions: {str(e)}")
        return {}

def extract_image_id(filename):
    """Extract COCO image ID from filename"""
    match = re.search(r'COCO_train2014_(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def get_libertine_font(size=80):
    """Try to load Libertine font with fallbacks"""
    system = platform.system()
    font_paths = []
    
    if system == "Windows":
        font_paths = [
            "C:/Windows/Fonts/timesbd.ttf",
            "C:/Windows/Fonts/times.ttf",
            "C:/Windows/Fonts/arial.ttf"
        ]
    elif system == "Darwin":
        font_paths = [
            "/Library/Fonts/Times New Roman Bold.ttf",
            "/Library/Fonts/Times New Roman.ttf",
            "/Library/Fonts/Arial.ttf"
        ]
    else:
        font_paths = [
            "/usr/share/fonts/TTF/DejaVuSerif-Bold.ttf",
            "/usr/share/fonts/TTF/DejaVuSerif.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf"
        ]
    
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    
    return ImageFont.load_default()

def find_step_folders(base_dir="data/coco", cfg=7):
    """Find all step variation folders for a given CFG value"""
    pattern = f"{base_dir}/sd_to_sd_cfg_{cfg}_steps_*_gen_0"  # Only look at gen_0 folders to avoid duplicates
    folders = glob.glob(pattern)
    step_counts = []
    
    for folder in folders:
        match = re.search(r'steps_(\d+)_gen_0', folder)
        if match:
            step_counts.append(int(match.group(1)))
    
    return sorted(list(set(step_counts)))  # Remove duplicates since each step has multiple generations

def create_comparison_grid(step_counts, generations=[0, 2, 4], cfg=7, indices=[84, 96]):
    """Create a grid comparing different steps and generations"""
    base_dir = "data/coco"
    
    # Load captions
    captions = load_coco_captions()
    
    # Get list of images from first folder to ensure consistency
    first_folder = f"{base_dir}/sd_to_sd_cfg_{cfg}_steps_{step_counts[0]}_gen_0"
    all_images = sorted([f for f in os.listdir(first_folder) if f.endswith(('.jpg', '.png'))])[:1000]  # Use first 1000 images
    print(f"Using {len(all_images)} images from the training set")
    
    # Use specified indices
    sample_images = []
    for idx in indices:
        if 0 <= idx < len(all_images):
            sample_images.append(all_images[idx])
        else:
            print(f"Warning: Index {idx} is out of range (0-{len(all_images)-1})")
    
    if not sample_images:
        print("No valid images found for the specified indices")
        return []
    
    # Load all images
    grid_data = []
    for img_file in sample_images:
        row_data = []
        image_id = extract_image_id(img_file)
        caption = captions[image_id][0] if image_id in captions else None
        
        # Group by generation first
        for gen in generations:
            step_images = []
            for steps in step_counts:
                folder = f"{base_dir}/sd_to_sd_cfg_{cfg}_steps_{steps}_gen_{gen}"
                img_path = os.path.join(folder, img_file)
                if os.path.exists(img_path):
                    try:
                        img = Image.open(img_path)
                        step_images.append((steps, img))
                    except Exception as e:
                        print(f"Error loading image {img_path}: {e}")
                        step_images.append((steps, None))
                else:
                    print(f"Missing image: {img_path}")
                    step_images.append((steps, None))
            row_data.append((gen, step_images))
        grid_data.append((caption, row_data))
    
    return grid_data

def create_grid_visualization(grid_data, output_path="step_generation_comparison.png"):
    """Create and save the grid visualization"""
    if not grid_data:
        print("No data to visualize")
        return
    
    # Get dimensions from first valid image
    for _, row_data in grid_data:
        for _, step_images in row_data:
            for _, img in step_images:
                if img is not None:
                    img_width, img_height = img.size
                    break
            if 'img_width' in locals():
                break
        if 'img_width' in locals():
            break
    
    if 'img_width' not in locals():
        print("No valid images found")
        return
    
    # Calculate grid dimensions
    n_rows = len(grid_data)  # number of sample images
    n_gen_groups = len(grid_data[0][1])  # number of generation groups
    n_steps = len(grid_data[0][1][0][1])  # number of step variations per generation
    
    # Add spacing
    spacing = 40
    caption_height = 150
    gen_label_height = 100
    step_label_height = 100
    
    # Calculate total dimensions
    total_width = (img_width * n_steps + spacing) * n_gen_groups - spacing
    total_height = (img_height + caption_height) * n_rows + gen_label_height + step_label_height
    
    # Create image
    grid_img = Image.new('RGB', (total_width, total_height), color='white')
    draw = ImageDraw.Draw(grid_img)
    
    # Get fonts
    gen_font = get_libertine_font(70)
    step_font = get_libertine_font(60)
    caption_font = get_libertine_font(50)
    
    # Draw the grid
    for row_idx, (caption, row_data) in enumerate(grid_data):
        y_offset = row_idx * (img_height + caption_height) + gen_label_height + step_label_height
        
        # Draw caption
        if caption:
            caption_y = y_offset + img_height + 10
            draw.text((10, caption_y), f'"{caption}"', fill="black", font=caption_font)
        
        for gen_idx, (gen, step_images) in enumerate(row_data):
            x_offset = gen_idx * (img_width * n_steps + spacing)
            
            # Draw generation label at the top
            if row_idx == 0:
                gen_text = f"Generation {gen}"
                draw.text((x_offset + (img_width * n_steps) // 2 - 100, 20), 
                         gen_text, fill="black", font=gen_font)
            
            # Draw step labels under generation label
            if row_idx == 0:
                for step_idx, (steps, _) in enumerate(step_images):
                    step_x = x_offset + step_idx * img_width
                    step_text = f"{steps}"
                    draw.text((step_x + img_width//2 - 20, gen_label_height + 10), 
                             step_text, fill="black", font=step_font)
            
            # Draw images for each step
            for step_idx, (steps, img) in enumerate(step_images):
                img_x = x_offset + step_idx * img_width
                
                if img is not None:
                    grid_img.paste(img, (img_x, y_offset))
                else:
                    # Draw placeholder for missing image
                    draw.rectangle([(img_x, y_offset), 
                                  (img_x + img_width, y_offset + img_height)], 
                                 outline="red", fill="lightgray")
                    draw.text((img_x + 10, y_offset + img_height//2), 
                            "Missing", fill="red", font=step_font)
    
    # Save the visualization
    grid_img.save(output_path, quality=95)
    print(f"Grid visualization saved to: {output_path}")
    
    # Display using matplotlib
    plt.figure(figsize=(20, 20))
    plt.imshow(np.array(grid_img))
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    # Find available step counts
    step_counts = find_step_folders()
    if not step_counts:
        print("No step variation folders found!")
        return
    
    print(f"Found step variations: {step_counts}")
    
    # Create comparison grid with specific indices
    grid_data = create_comparison_grid(step_counts, generations=[0, 2, 4], cfg=7, indices=[84, 96])
    
    # Create and save visualization
    create_grid_visualization(grid_data, "step_generation_comparison.png")

if __name__ == "__main__":
    main() 