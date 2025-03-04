import os
import re
import json
import random
import argparse
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import textwrap
import glob

def load_coco_captions(annotation_file="data/coco/annotations/captions_train2014.json"):
    """Load COCO captions from annotation file"""
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # Create mapping from image ID to captions
    image_id_to_caption = {}
    for annotation in annotations['annotations']:
        image_id = annotation['image_id']
        caption = annotation['caption']
        
        if image_id not in image_id_to_caption:
            image_id_to_caption[image_id] = []
        image_id_to_caption[image_id].append(caption)
    
    return image_id_to_caption

def extract_image_id(filename):
    """Extract image ID from filename"""
    match = re.search(r'COCO_train2014_0*(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def find_step_folders(base_dir="data/coco", cfg=7, gen=0):
    """Find folders for different step counts"""
    pattern = f"{base_dir}/sd_to_sd_cfg_{cfg}_steps_*_gen_{gen}"
    folders = glob.glob(pattern)
    
    step_folders = []
    for folder in folders:
        match = re.search(r'steps_(\d+)_gen', folder)
        if match:
            step = int(match.group(1))
            step_folders.append((step, folder))
    
    return sorted(step_folders)

def create_comparison_grid(step_counts, generations, cfg=7, indices=[84, 96]):
    """Create a grid of images comparing different step counts across generations"""
    # Load COCO captions
    image_id_to_caption = load_coco_captions()
    
    # Initialize grid data
    grid_data = {
        'generations': [],
        'step_counts': step_counts,
        'images': {},
        'captions': {}
    }
    
    # For each generation
    for gen in generations:
        gen_data = {'gen': gen, 'folders': []}
        
        # Find folders for each step count
        for step in step_counts:
            folder_path = f"data/coco/sd_to_sd_cfg_{cfg}_steps_{step}_gen_{gen}"
            if os.path.exists(folder_path):
                gen_data['folders'].append((step, folder_path))
            else:
                print(f"Warning: Folder not found: {folder_path}")
        
        if gen_data['folders']:
            grid_data['generations'].append(gen_data)
    
    # For each image index
    for idx in indices:
        grid_data['images'][idx] = {}
        
        # Get all images for this index across generations and steps
        for gen_data in grid_data['generations']:
            gen = gen_data['gen']
            grid_data['images'][idx][gen] = {}
            
            for step, folder in gen_data['folders']:
                # Find image with this index in the folder
                image_files = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.png'))]
                sorted_files = sorted(image_files)
                
                if idx < len(sorted_files):
                    image_file = sorted_files[idx]
                    image_path = os.path.join(folder, image_file)
                    
                    # Get caption for this image
                    image_id = extract_image_id(image_file)
                    caption = ""
                    if image_id and image_id in image_id_to_caption:
                        caption = image_id_to_caption[image_id][0]  # Use first caption
                    
                    grid_data['images'][idx][gen][step] = image_path
                    if idx not in grid_data['captions']:
                        grid_data['captions'][idx] = caption
    
    return grid_data

def create_grid_visualization(grid_data, output_path="step_generation_comparison_by_gen.png"):
    """Create a visualization using matplotlib for reliable text rendering"""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    import numpy as np
    
    # Get dimensions
    num_generations = len(grid_data['generations'])
    num_steps = len(grid_data['step_counts'])
    num_indices = len(grid_data['images'].keys())
    
    # Create figure with custom size - balanced approach
    fig_width = 4 + (3 * num_steps)  # Balanced width
    fig_height = 1.5 + (3 * num_generations)  # Balanced height
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Create grid layout with reasonable spacing
    gs = gridspec.GridSpec(num_generations + 1, num_steps + 1,
                          width_ratios=[0.6] + [1] * num_steps,
                          height_ratios=[0.2] + [1] * num_generations,
                          wspace=0.1,  # Reasonable spacing
                          hspace=0.1)
    
    # Function to add an image to the plot
    def add_image_to_plot(ax, img_path):
        img = plt.imread(img_path)
        ax.imshow(img)
        ax.axis('off')
    
    # Add step labels at the top
    for k, step in enumerate(grid_data['step_counts']):
        ax = plt.subplot(gs[0, k+1])
        ax.text(0.5, 0, f"{step} steps", 
                fontsize=32,  # Large text
                fontweight='bold',
                ha='center', va='center')
        ax.axis('off')
    
    # For each image index (should be just one)
    for i, idx in enumerate(grid_data['images'].keys()):
        # For each generation
        for j, gen_data in enumerate(grid_data['generations']):
            gen = gen_data['gen']
            
            # Add generation label on the left
            ax = plt.subplot(gs[j+1, 0])
            ax.text(0.5, 0.5, f"Gen {gen}", 
                    fontsize=32,  # Large text
                    fontweight='bold',
                    ha='center', va='center')
            ax.axis('off')
            
            # For each step
            for k, step in enumerate(grid_data['step_counts']):
                ax = plt.subplot(gs[j+1, k+1])
                
                # Add image if available
                if gen in grid_data['images'][idx] and step in grid_data['images'][idx][gen]:
                    image_path = grid_data['images'][idx][gen][step]
                    add_image_to_plot(ax, image_path)
                else:
                    ax.text(0.5, 0.5, "N/A", ha='center', va='center')
                    ax.axis('off')
    
    # Adjust layout - balanced approach
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Saved comparison grid to {output_path}")

def main():
    # Define parameters
    cfg = 7
    step_counts = [10, 20, 50, 500]
    generations = [0, 10]  # Only use generations 0 and 10
    indices = [84]  # Use index 84 (the first of the two indices previously used)
    
    # Create comparison grid
    print(f"Creating comparison grid for CFG {cfg}, steps {step_counts}, generations {generations}...")
    grid_data = create_comparison_grid(step_counts, generations, cfg, indices)
    
    # Create visualization
    create_grid_visualization(grid_data)
    print("Done!")

if __name__ == "__main__":
    main() 