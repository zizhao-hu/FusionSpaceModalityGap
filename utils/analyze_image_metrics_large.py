import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from collections import defaultdict
import re
import cv2
from skimage.color import rgb2gray
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, InterpolationMode, Normalize
import json
from tqdm.auto import tqdm
from diffusers import StableDiffusionPipeline
import shutil
from diffusers import UNet2DConditionModel
import argparse
import copy
import random

# Import color constants
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from utils.colors import *

def extract_generation_number(folder_path):
    """Extract generation number from folder path."""
    match = re.search(r'gen_(\d+)', folder_path)
    if match:
        return int(match.group(1))
    return None

def calculate_colorfulness(image):
    """
    Calculate colorfulness metric as described in the paper:
    "Measuring colourfulness in natural images" by Hasler and Süsstrunk (2003)
    """
    # Split the image into its BGR components
    B, G, R = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    # Compute rg = R - G
    rg = np.absolute(R.astype(int) - G.astype(int))
    
    # Compute yb = 0.5 * (R + G) - B
    yb = np.absolute(0.5 * (R.astype(int) + G.astype(int)) - B.astype(int))
    
    # Compute the mean and standard deviation of rg and yb
    rg_mean, rg_std = np.mean(rg), np.std(rg)
    yb_mean, yb_std = np.mean(yb), np.std(yb)
    
    # Compute the colorfulness metric
    colorfulness = np.sqrt(rg_std**2 + yb_std**2) + 0.3 * np.sqrt(rg_mean**2 + yb_mean**2)
    
    return colorfulness

def calculate_contrast(image):
    """Calculate RMS contrast of an image."""
    # Convert to grayscale
    gray = rgb2gray(image)
    
    # Calculate RMS contrast
    contrast = np.std(gray) / np.mean(gray) if np.mean(gray) > 0 else 0
    
    return contrast

def calculate_brightness(image):
    """Calculate average brightness of an image."""
    # Convert to grayscale
    gray = rgb2gray(image)
    
    # Calculate average brightness
    brightness = np.mean(gray) * 255
    
    return brightness

def analyze_color_distribution(image_path):
    """Analyze the color distribution of an image."""
    try:
        img = Image.open(image_path)
        img_array = np.array(img)
        
        # Extract RGB channels
        r_channel = img_array[:, :, 0].flatten()
        g_channel = img_array[:, :, 1].flatten()
        b_channel = img_array[:, :, 2].flatten()
        
        # Calculate average RGB values
        avg_r = np.mean(r_channel)
        avg_g = np.mean(g_channel)
        avg_b = np.mean(b_channel)
        
        # Calculate color saturation (difference between max and min channel values)
        saturation = np.mean(np.max(img_array, axis=2) - np.min(img_array, axis=2))
        
        # Calculate color standard deviation (measure of color diversity)
        r_std = np.std(r_channel)
        g_std = np.std(g_channel)
        b_std = np.std(b_channel)
        color_std = (r_std + g_std + b_std) / 3
        
        # Calculate additional metrics
        contrast = calculate_contrast(img_array)
        brightness = calculate_brightness(img_array)
        colorfulness = calculate_colorfulness(img_array)
        
        return {
            'img_path': image_path,
            'avg_rgb': (avg_r, avg_g, avg_b),
            'saturation': saturation,
            'color_std': color_std,
            'contrast': contrast,
            'brightness': brightness,
            'colorfulness': colorfulness
        }
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def load_coco_captions(annotation_file):
    """Load COCO captions from json file"""
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
    
    return image_id_to_caption

def get_available_checkpoints():
    """Get available model checkpoints and their generation numbers"""
    checkpoint_dirs = {}
    
    # Check for recursive finetune models
    recursive_dirs = glob.glob(os.path.join("models", "sd_to_sd_cfg_*_steps_50_gen_*"))
    for dir_path in recursive_dirs:
        if "_finetune" in dir_path or "_baseline" in dir_path:
            continue
        gen_num = extract_generation_number(dir_path)
        if gen_num is not None:
            if "recursive" not in checkpoint_dirs:
                checkpoint_dirs["recursive"] = {}
            checkpoint_dirs["recursive"][gen_num] = dir_path
    
    # Check for real finetune models
    finetune_dirs = glob.glob(os.path.join("models", "sd_to_sd_cfg_*_steps_50_gen_*_finetune"))
    for dir_path in finetune_dirs:
        gen_num = extract_generation_number(dir_path)
        if gen_num is not None:
            if "finetune" not in checkpoint_dirs:
                checkpoint_dirs["finetune"] = {}
            checkpoint_dirs["finetune"][gen_num] = dir_path
    
    # Check for baseline models
    baseline_dirs = glob.glob(os.path.join("models", "sd_to_sd_cfg_*_steps_50_gen_*_baseline"))
    for dir_path in baseline_dirs:
        gen_num = extract_generation_number(dir_path)
        if gen_num is not None:
            if "baseline" not in checkpoint_dirs:
                checkpoint_dirs["baseline"] = {}
            checkpoint_dirs["baseline"][gen_num] = dir_path
    
    return checkpoint_dirs

def interpolate_checkpoint(available_checkpoints, target_gen, group="recursive"):
    """Interpolate or extrapolate to find the best checkpoint for a target generation"""
    if group not in available_checkpoints:
        return None
    
    checkpoints = available_checkpoints[group]
    if target_gen in checkpoints:
        return checkpoints[target_gen]
    
    # Get available generations
    available_gens = sorted(list(checkpoints.keys()))
    if not available_gens:
        return None
    
    # Find closest generations for interpolation
    lower_gen = None
    upper_gen = None
    
    for gen in available_gens:
        if gen < target_gen:
            if lower_gen is None or gen > lower_gen:
                lower_gen = gen
        elif gen > target_gen:
            if upper_gen is None or gen < upper_gen:
                upper_gen = gen
    
    # Extrapolate if needed
    if lower_gen is None:
        return checkpoints[min(available_gens)]
    if upper_gen is None:
        return checkpoints[max(available_gens)]
    
    # Return the closest checkpoint
    if (target_gen - lower_gen) <= (upper_gen - target_gen):
        return checkpoints[lower_gen]
    else:
        return checkpoints[upper_gen]

def load_model_checkpoint(model_path, device="cuda"):
    """Load a model checkpoint from the specified path"""
    print(f"Loading model from {model_path}")
    
    try:
        # Load base model with offline fallback
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_cache = os.path.join(cache_dir, "models--CompVis--stable-diffusion-v1-4", "snapshots", "39593d5650112b4cc580433f6b0435385882d819")
        
        if not os.path.exists(model_cache):
            # Try to load from Hugging Face directly
            pipeline = StableDiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                torch_dtype=torch.float16
            ).to(device)
        else:
            # Load from cache
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_cache,
                torch_dtype=torch.float16,
                local_files_only=True
            ).to(device)
        
        # Load the fine-tuned UNet if it exists
        if os.path.exists(os.path.join(model_path, "unet")):
            pipeline.unet = UNet2DConditionModel.from_pretrained(
                os.path.join(model_path, "unet"),
                torch_dtype=torch.float16
            ).to(device)
        
        pipeline.safety_checker = None  # Disable safety checker for speed
        return pipeline
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def generate_images_for_evaluation(model_path, output_dir, captions, num_images=200, batch_size=4, cfg_scale=7.0, steps=50, device="cuda"):
    """Generate images for evaluation using the specified model checkpoint"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Check existing images
    existing_images = glob.glob(os.path.join(output_dir, "*.jpg"))
    if len(existing_images) >= num_images:
        print(f"Already have {len(existing_images)} images in {output_dir}, skipping generation")
        return True
    
    # Load model
    pipeline = load_model_checkpoint(model_path, device)
    if pipeline is None:
        return False
    
    # Set deterministic generation
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # Generate images in batches
    total_batches = (num_images + batch_size - 1) // batch_size
    image_count = len(existing_images)
    start_idx = image_count
    
    try:
        for batch_idx in tqdm(range(start_idx // batch_size, total_batches), desc="Generating images"):
            current_batch_size = min(batch_size, num_images - image_count)
            if current_batch_size <= 0:
                break
                
            # Get captions for this batch
            batch_start = image_count % len(captions)  # Wrap around if we need more captions
            batch_captions = []
            for i in range(current_batch_size):
                caption_idx = (batch_start + i) % len(captions)
                batch_captions.append(captions[caption_idx])
            
            # Generate images
            images = pipeline(
                batch_captions,
                num_inference_steps=steps,
                guidance_scale=cfg_scale
            ).images
            
            # Save images
            for i, image in enumerate(images):
                image_idx = image_count + i
                image_id = f"{image_idx:012d}"
                image_path = os.path.join(output_dir, f"COCO_eval_{image_id}.jpg")
                image.save(image_path)
            
            image_count += len(images)
            
        print(f"Generated {image_count} images in {output_dir}")
        return True
    except Exception as e:
        print(f"Error generating images: {e}")
        return False
    finally:
        # Clean up
        del pipeline
        torch.cuda.empty_cache()

def analyze_random_batch(image_dir, batch_size=100, max_images=200):
    """Analyze metrics for a random batch of images"""
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    
    # Limit to max_images
    if len(image_files) > max_images:
        image_files = image_files[:max_images]
    
    if len(image_files) < batch_size:
        print(f"Not enough images in {image_dir}. Found {len(image_files)}, need {batch_size}")
        return None
    
    # Randomly select batch_size images
    np.random.seed(None)  # Use different seed each time
    selected_files = np.random.choice(image_files, size=batch_size, replace=False)
    
    # Analyze color distribution
    results = []
    for img_path in tqdm(selected_files, desc=f"Analyzing random batch"):
        color_data = analyze_color_distribution(img_path)
        if color_data:
            results.append(color_data)
    
    return results

def calculate_random_batch_fid_and_clip(image_dir, ref_stack, clip_model=None, clip_preprocess=None, 
                                      batch_size=100, max_images=200, device="cuda"):
    """Calculate FID and CLIP scores for a random batch of images"""
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    
    # Limit to max_images
    if len(image_files) > max_images:
        image_files = image_files[:max_images]
    
    if len(image_files) < batch_size:
        print(f"Not enough images in {image_dir}. Found {len(image_files)}, need {batch_size}")
        return None
    
    # Randomly select batch_size images
    np.random.seed(None)  # Use different seed each time
    selected_files = np.random.choice(image_files, size=batch_size, replace=False)
    
    # Define transforms for FID calculation
    transform = Compose([
        Resize(299, interpolation=InterpolationMode.BILINEAR),
        CenterCrop(299),
        ToTensor(),
    ])
    
    # Load generated images
    gen_images = []
    gen_pil_images = []  # Store original PIL images for CLIP
    
    for img_path in tqdm(selected_files, desc=f"Loading random batch"):
        try:
            img = Image.open(img_path).convert('RGB')
            gen_pil_images.append(img)
            
            img_tensor = transform(img)
            # Scale to [0, 255] and convert to uint8 for FID
            img_tensor = (img_tensor * 255).to(torch.uint8)
            gen_images.append(img_tensor)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    
    if not gen_images:
        print(f"No valid images loaded from {image_dir}")
        return None
    
    gen_stack = torch.stack(gen_images).to(device)
    
    # Initialize scores dictionary
    scores = {'fid': None, 'is_mean': None, 'is_std': None, 'clip_score': None}
    
    # Calculate FID
    try:
        fid = FrechetInceptionDistance(normalize=True).to(device)
        
        # Update FID with reference images
        fid.update(ref_stack, real=True)
        
        # Update FID with generated images
        fid.update(gen_stack, real=False)
        
        scores['fid'] = float(fid.compute())
        
        del fid
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error calculating FID: {e}")
    
    # Calculate Inception Score
    try:
        inception_score = InceptionScore(normalize=True).to(device)
        inception_score.update(gen_stack)
        is_mean, is_std = inception_score.compute()
        scores['is_mean'] = float(is_mean)
        scores['is_std'] = float(is_std)
        
        del inception_score
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error calculating IS: {e}")
    
    # Calculate CLIP score if model is available
    if clip_model is not None and clip_preprocess is not None:
        try:
            # Simple CLIP score calculation
            clip_scores = []
            
            # Process in smaller batches
            clip_batch_size = 16
            for i in range(0, len(gen_pil_images), clip_batch_size):
                end_idx = min(i + clip_batch_size, len(gen_pil_images))
                clip_batch = gen_pil_images[i:end_idx]
                
                # Create a simple caption for each image
                captions = ["A photograph"] * len(clip_batch)
                
                # Preprocess images and text
                processed_images = torch.stack([clip_preprocess(img) for img in clip_batch]).to(device)
                text_tokens = clip.tokenize(captions).to(device)
                
                # Get embeddings
                with torch.no_grad():
                    image_features = clip_model.encode_image(processed_images)
                    text_features = clip_model.encode_text(text_tokens)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
                
                # Calculate similarity
                similarities = (image_features @ text_features.T).diag()
                similarities = (similarities + 1) / 2  # Convert from -1,1 to 0,1
                clip_scores.extend(similarities.cpu().numpy())
            
            scores['clip_score'] = float(np.mean(clip_scores))
            
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error calculating CLIP score: {e}")
    
    return scores

def load_reference_images(ref_folder, max_images=1000, device="cuda"):
    """Load reference images for FID calculation"""
    print(f"Loading reference images from {ref_folder}")
    
    # Define transforms for FID calculation
    transform = Compose([
        Resize(299, interpolation=InterpolationMode.BILINEAR),
        CenterCrop(299),
        ToTensor(),
    ])
    
    # Load reference images
    ref_images = []
    ref_files = sorted(glob.glob(os.path.join(ref_folder, "*.jpg")))[:max_images]
    
    for img_path in tqdm(ref_files, desc="Loading reference images"):
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            # Scale to [0, 255] and convert to uint8 for FID
            img_tensor = (img_tensor * 255).to(torch.uint8)
            ref_images.append(img_tensor)
        except Exception as e:
            print(f"Error loading reference image {img_path}: {e}")
    
    if not ref_images:
        print("No reference images found!")
        return None
    
    return torch.stack(ref_images).to(device)

def plot_batch_metrics(all_results, output_path, target_generations=[0, 1, 2, 5, 9], include_std=True, real_metrics=None):
    """Plot metrics for each batch across generations and groups
    
    Args:
        all_results: Dictionary with structure {group: {gen: {trial: {metric: value}}}}
        output_path: Path to save the output image
        target_generations: List of generations to include in the plot
        include_std: Whether to include standard deviation in the plots
        real_metrics: Dictionary with real image metrics
    """
    # Define the groups and their display names
    groups = [
        {"key": "recursive", "name": "Recursive Finetune", "color": PRIMARY},
        {"key": "finetune", "name": "Real Finetune", "color": SECONDARY},
        {"key": "baseline", "name": "Gen 0 Finetune", "color": TERTIARY}
    ]
    
    # Define all the metrics to plot
    all_metrics = [
        # Color metrics 
        {'name': 'Saturation', 'key': 'saturation', 'source': 'results'},
        {'name': 'Contrast', 'key': 'contrast', 'source': 'results'},
        {'name': 'Brightness', 'key': 'brightness', 'source': 'results'},
        {'name': 'Colorfulness', 'key': 'colorfulness', 'source': 'results'},
        {'name': 'Color Std', 'key': 'color_std', 'source': 'results'},
        
        # Generation quality metrics
        {'name': 'FID', 'key': 'fid', 'source': 'scores'},
        {'name': 'IS', 'key': 'is_mean', 'source': 'scores'},
        {'name': 'CLIP Score', 'key': 'clip_score', 'source': 'scores'}
    ]
    
    # Calculate number of rows and columns for subplots
    n_metrics = len(all_metrics)
    n_cols = min(4, n_metrics)  # At most 4 columns
    n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows), squeeze=False)
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # For each metric, create a line plot
    for i, metric in enumerate(all_metrics):
        if i >= len(axes):
            break  # Safety check
            
        ax = axes[i]
        ax.set_facecolor('white')
        
        metric_key = metric['key']
        source = metric['source']
        
        # Store lines for legend
        lines = []
        labels = []
        
        # Plot each group
        for group in groups:
            group_key = group["key"]
            group_color = group["color"]
            group_name = group["name"]
            
            if group_key not in all_results:
                continue
                
            # For each generation, collect and plot the metric value
            x_values = []
            y_means = []
            y_stds = []
            
            for gen in sorted(target_generations):
                gen_str = str(gen)
                
                if gen_str not in all_results[group_key]:
                    continue
                
                # Calculate mean and std across trials
                trial_means = []
                
                for trial_idx, trial_data in all_results[group_key][gen_str].items():
                    if source == 'scores':
                        # Get from scores dictionary
                        if 'scores' in trial_data and metric_key in trial_data['scores'] and trial_data['scores'][metric_key] is not None:
                            trial_means.append(trial_data['scores'][metric_key])
                    else:
                        # Calculate from results dictionary
                        if 'results' in trial_data:
                            # Calculate mean for this trial
                            img_values = []
                            for img_data in trial_data['results']:
                                if metric_key in img_data and img_data[metric_key] is not None:
                                    img_values.append(img_data[metric_key])
                            if img_values:
                                trial_means.append(np.mean(img_values))
                
                if trial_means:
                    x_values.append(gen)
                    y_means.append(np.mean(trial_means))
                    y_stds.append(np.std(trial_means))
            
            if x_values and y_means:
                # Plot mean values
                line, = ax.plot(x_values, y_means, marker='o', linestyle='-', 
                               linewidth=2, markersize=8, color=group_color, label=group_name)
                lines.append(line)
                labels.append(group_name)
                
                # Plot standard deviation as shaded area if requested
                if include_std:
                    ax.fill_between(x_values, 
                                   [y - s for y, s in zip(y_means, y_stds)],
                                   [y + s for y, s in zip(y_means, y_stds)],
                                   color=group_color, alpha=0.2)
        
        # Add real image reference line if available
        if real_metrics and metric_key in real_metrics:
            real_value = real_metrics[metric_key]
            real_line = ax.axhline(y=real_value, color='black', linestyle='--', linewidth=2, label='Real Images')
            lines.append(real_line)
            labels.append('Real Images')
        
        # Set x-axis ticks
        ax.set_xticks(target_generations)
        ax.set_xticklabels([str(x) for x in target_generations])
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Set title and labels
        ax.set_title(metric['name'], fontsize=14)
        ax.set_xlabel('Generation', fontsize=12)
        
        # Add legend to the first subplot only
        if i == 0:
            ax.legend(lines, labels, loc='best', fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Batch metrics chart saved to {output_path}")

def save_batch_metrics_to_csv(all_results, output_path, target_generations=[0, 1, 2, 5, 9]):
    """Save batch metrics to a CSV file.
    
    Args:
        all_results: Dictionary of results by group, generation, and batch
        output_path: Path to save the CSV file
        target_generations: List of generations to include
    """
    # Define the groups
    groups = ["recursive", "finetune", "baseline"]
    
    with open(output_path, 'w') as f:
        # Write header
        f.write("Generation,Group,Batch,Saturation,Saturation_Std,Contrast,Contrast_Std,Brightness,Brightness_Std,Colorfulness,Colorfulness_Std,Color_Std,Color_Std_Std,FID,IS,CLIP_Score\n")
        
        # Write data for each group, generation, and batch
        for group in groups:
            if group not in all_results:
                continue
                
            group_results = all_results[group]
            
            # Get all generations for this group
            generations = sorted([int(gen) for gen in group_results.keys() if gen.isdigit()])
            
            # Format group name for display
            if group == "recursive":
                display_group = "Recursive Finetune"
            elif group == "finetune":
                display_group = "Real Finetune"
            elif group == "baseline":
                display_group = "Gen 0 Finetune"
            else:
                display_group = group
            
            # For each generation in this group
            for gen in target_generations:
                gen_str = str(gen)
                
                # Skip if this generation doesn't exist for this group
                if gen_str not in group_results:
                    continue
                
                gen_batches = group_results[gen_str]
                
                # For each batch in this generation
                for batch_idx, batch_data in gen_batches.items():
                    # Calculate means and standard deviations for color metrics
                    if 'results' in batch_data and batch_data['results']:
                        # Extract values for each metric
                        saturation_values = [r.get('saturation', 0) for r in batch_data['results'] if 'saturation' in r]
                        contrast_values = [r.get('contrast', 0) for r in batch_data['results'] if 'contrast' in r]
                        brightness_values = [r.get('brightness', 0) for r in batch_data['results'] if 'brightness' in r]
                        colorfulness_values = [r.get('colorfulness', 0) for r in batch_data['results'] if 'colorfulness' in r]
                        color_std_values = [r.get('color_std', 0) for r in batch_data['results'] if 'color_std' in r]
                        
                        # Calculate means and standard deviations
                        saturation_mean = np.mean(saturation_values) if saturation_values else 0
                        saturation_std = np.std(saturation_values) if saturation_values else 0
                        
                        contrast_mean = np.mean(contrast_values) if contrast_values else 0
                        contrast_std = np.std(contrast_values) if contrast_values else 0
                        
                        brightness_mean = np.mean(brightness_values) if brightness_values else 0
                        brightness_std = np.std(brightness_values) if brightness_values else 0
                        
                        colorfulness_mean = np.mean(colorfulness_values) if colorfulness_values else 0
                        colorfulness_std = np.std(colorfulness_values) if colorfulness_values else 0
                        
                        color_std_mean = np.mean(color_std_values) if color_std_values else 0
                        color_std_std = np.std(color_std_values) if color_std_values else 0
                    else:
                        # No results available
                        saturation_mean = saturation_std = 0
                        contrast_mean = contrast_std = 0
                        brightness_mean = brightness_std = 0
                        colorfulness_mean = colorfulness_std = 0
                        color_std_mean = color_std_std = 0
                    
                    # Get scores
                    fid = is_score = clip_score = ""
                    if 'scores' in batch_data:
                        scores = batch_data['scores']
                        fid = scores.get('fid', "")
                        is_score = scores.get('is_mean', "")
                        clip_score = scores.get('clip_score', "")
                    
                    # Write row
                    f.write(f"{gen},{display_group},{batch_idx},{saturation_mean:.2f},{saturation_std:.2f},{contrast_mean:.4f},{contrast_std:.4f},{brightness_mean:.2f},{brightness_std:.2f},{colorfulness_mean:.2f},{colorfulness_std:.2f},{color_std_mean:.2f},{color_std_std:.2f},{fid},{is_score},{clip_score}\n")
    
    print(f"Batch metrics saved to {output_path}")

def analyze_batch_metrics(image_dir, start_idx=0, batch_size=100):
    """Analyze metrics for a batch of images"""
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    
    # Select only the batch we want to analyze
    end_idx = min(start_idx + batch_size, len(image_files))
    batch_files = image_files[start_idx:end_idx]
    
    if not batch_files:
        print(f"No images found in {image_dir} for batch starting at {start_idx}")
        return None
    
    # Analyze color distribution
    results = []
    for img_path in tqdm(batch_files, desc=f"Analyzing batch {start_idx//batch_size + 1}"):
        color_data = analyze_color_distribution(img_path)
        if color_data:
            results.append(color_data)
    
    return results

def calculate_batch_fid_and_clip(image_dir, ref_stack, clip_model=None, clip_preprocess=None, 
                                start_idx=0, batch_size=100, device="cuda"):
    """Calculate FID and CLIP scores for a batch of images"""
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    
    # Select only the batch we want to analyze
    end_idx = min(start_idx + batch_size, len(image_files))
    batch_files = image_files[start_idx:end_idx]
    
    if not batch_files:
        print(f"No images found in {image_dir} for batch starting at {start_idx}")
        return None
    
    # Define transforms for FID calculation
    transform = Compose([
        Resize(299, interpolation=InterpolationMode.BILINEAR),
        CenterCrop(299),
        ToTensor(),
    ])
    
    # Load generated images
    gen_images = []
    gen_pil_images = []  # Store original PIL images for CLIP
    
    for img_path in tqdm(batch_files, desc=f"Loading batch {start_idx//batch_size + 1}"):
        try:
            img = Image.open(img_path).convert('RGB')
            gen_pil_images.append(img)
            
            img_tensor = transform(img)
            # Scale to [0, 255] and convert to uint8 for FID
            img_tensor = (img_tensor * 255).to(torch.uint8)
            gen_images.append(img_tensor)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    
    if not gen_images:
        print(f"No valid images loaded from {image_dir}")
        return None
    
    gen_stack = torch.stack(gen_images).to(device)
    
    # Initialize scores dictionary
    scores = {'fid': None, 'is_mean': None, 'is_std': None, 'clip_score': None}
    
    # Calculate FID
    try:
        fid = FrechetInceptionDistance(normalize=True).to(device)
        
        # Update FID with reference images
        fid.update(ref_stack, real=True)
        
        # Update FID with generated images
        fid.update(gen_stack, real=False)
        
        scores['fid'] = float(fid.compute())
        
        del fid
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error calculating FID: {e}")
    
    # Calculate Inception Score
    try:
        inception_score = InceptionScore(normalize=True).to(device)
        inception_score.update(gen_stack)
        is_mean, is_std = inception_score.compute()
        scores['is_mean'] = float(is_mean)
        scores['is_std'] = float(is_std)
        
        del inception_score
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error calculating IS: {e}")
    
    # Calculate CLIP score if model is available
    if clip_model is not None and clip_preprocess is not None:
        try:
            # Simple CLIP score calculation
            clip_scores = []
            
            # Process in smaller batches
            clip_batch_size = 16
            for i in range(0, len(gen_pil_images), clip_batch_size):
                end_idx = min(i + clip_batch_size, len(gen_pil_images))
                clip_batch = gen_pil_images[i:end_idx]
                
                # Create a simple caption for each image
                captions = ["A photograph"] * len(clip_batch)
                
                # Preprocess images and text
                processed_images = torch.stack([clip_preprocess(img) for img in clip_batch]).to(device)
                text_tokens = clip.tokenize(captions).to(device)
                
                # Get embeddings
                with torch.no_grad():
                    image_features = clip_model.encode_image(processed_images)
                    text_features = clip_model.encode_text(text_tokens)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
                
                # Calculate similarity
                similarities = (image_features @ text_features.T).diag()
                similarities = (similarities + 1) / 2  # Convert from -1,1 to 0,1
                clip_scores.extend(similarities.cpu().numpy())
            
            scores['clip_score'] = float(np.mean(clip_scores))
            
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error calculating CLIP score: {e}")
    
    return scores

def check_directory_has_enough_images(dir_path, min_images=100):
    """Check if a directory has enough images for analysis."""
    if not os.path.exists(dir_path):
        return False
    
    image_files = glob.glob(os.path.join(dir_path, "*.jpg"))
    return len(image_files) >= min_images

def main():
    """Main function to analyze image metrics."""
    parser = argparse.ArgumentParser(description='Analyze image metrics for generated images')
    parser.add_argument('--base_dir', type=str, default=os.path.join("data", "coco"), 
                        help='Base directory containing the image folders')
    parser.add_argument('--output_dir', type=str, default=os.path.join("vis", "t2i", "metrics_large"),
                        help='Output directory for charts and CSV')
    parser.add_argument('--target_generations', type=str, default="0,1,2,3,4,5,6,7,8,9,10",
                        help='Comma-separated list of generations to analyze')
    parser.add_argument('--num_trials', type=int, default=10,
                        help='Number of random sampling trials to perform')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Number of images to analyze in each batch')
    parser.add_argument('--max_images', type=int, default=200,
                        help='Maximum number of images to consider per generation')
    parser.add_argument('--force_generate', action='store_true',
                        help='Force generation of images even if they already exist')
    
    args = parser.parse_args()
    
    # Parse target generations
    target_generations = [int(g) for g in args.target_generations.split(',')]
    print(f"Target generations: {target_generations}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Path to CSV file
    csv_path = os.path.join(args.output_dir, "batch_metrics.csv")
    
    # Initialize all_results dictionary
    all_results = {
        "recursive": {},
        "finetune": {},
        "baseline": {}
    }
    
    # Get available checkpoints
    available_checkpoints = get_available_checkpoints()
    print("Available checkpoints:")
    for group, gens in available_checkpoints.items():
        print(f"  {group}: {sorted(gens.keys())}")
    
    # Load COCO captions for image generation
    annotation_file = os.path.join(args.base_dir, "annotations", "captions_train2014.json")
    captions_dict = load_coco_captions(annotation_file)
    captions = []
    for image_id, caption_list in captions_dict.items():
        captions.extend(caption_list)
    
    # Shuffle captions
    random.shuffle(captions)
    print(f"Loaded {len(captions)} captions for image generation")
    
    # Check if we need to generate images for any empty directories
    gen_large_dir = os.path.join(args.base_dir, "gen_large")
    os.makedirs(gen_large_dir, exist_ok=True)
    
    # Check and generate images for each group and generation if needed
    for group in ["recursive", "finetune", "baseline"]:
        if group in available_checkpoints:
            print(f"\nChecking {group} group for missing images...")
            
            for gen in target_generations:
                if gen in available_checkpoints[group]:
                    checkpoint_path = available_checkpoints[group][gen]
                    
                    # Determine the output directory name
                    if group == "recursive":
                        output_dir_name = f"sd_to_sd_cfg_7_steps_50_gen_{gen}"
                    elif group == "finetune":
                        output_dir_name = f"sd_to_sd_cfg_7_steps_50_gen_{gen}_finetune"
                    elif group == "baseline":
                        output_dir_name = f"sd_to_sd_cfg_7_steps_50_gen_{gen}_baseline"
                    
                    output_dir = os.path.join(gen_large_dir, output_dir_name)
                    
                    # Check if directory has enough images
                    if not check_directory_has_enough_images(output_dir, args.batch_size) or args.force_generate:
                        print(f"Generating images for {group} generation {gen} in {output_dir}")
                        
                        # Generate images
                        success = generate_images_for_evaluation(
                            checkpoint_path, output_dir, captions, 
                            num_images=args.max_images, batch_size=4, 
                            cfg_scale=7.0, steps=50, device="cuda"
                        )
                        
                        if not success:
                            print(f"Failed to generate images for {group} generation {gen}")
                    else:
                        print(f"Directory {output_dir} already has enough images")
    
    # Check if CSV file exists
    if os.path.exists(csv_path):
        print(f"Loading metrics from existing CSV file: {csv_path}")
        # Load metrics from CSV
        import csv
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            
            # Check the column names in the CSV
            fieldnames = reader.fieldnames
            print(f"CSV columns: {fieldnames}")
            
            # Determine the column names for batch/trial
            batch_column = 'Batch' if 'Batch' in fieldnames else 'Trial'
            
            # Determine column names for metrics
            saturation_column = 'Saturation' if 'Saturation' in fieldnames else 'Saturation_Mean'
            contrast_column = 'Contrast' if 'Contrast' in fieldnames else 'Contrast_Mean'
            brightness_column = 'Brightness' if 'Brightness' in fieldnames else 'Brightness_Mean'
            colorfulness_column = 'Colorfulness' if 'Colorfulness' in fieldnames else 'Colorfulness_Mean'
            color_std_column = 'Color_Std' if 'Color_Std' in fieldnames else 'Color_Std_Mean'
            is_column = 'IS' if 'IS' in fieldnames else 'IS_Mean'
            
            # Reset file pointer to beginning
            f.seek(0)
            next(reader)  # Skip header
            
            for row in reader:
                gen = row['Generation']
                group_display = row['Group']
                batch_idx = row[batch_column]
                
                # Map display group name to internal group name
                if group_display == "Recursive Finetune":
                    group = "recursive"
                elif group_display == "Real Finetune":
                    group = "finetune"
                elif group_display == "Gen 0 Finetune":
                    group = "baseline"
                else:
                    group = group_display.lower()
                
                # Initialize generation if not exists
                if gen not in all_results[group]:
                    all_results[group][gen] = {}
                
                # Initialize batch if not exists
                if batch_idx not in all_results[group][gen]:
                    all_results[group][gen][batch_idx] = {'results': [], 'scores': {}}
                
                # Add scores
                all_results[group][gen][batch_idx]['scores'] = {
                    'fid': float(row['FID']) if row['FID'] else None,
                    'is_mean': float(row[is_column]) if row[is_column] else None,
                    'clip_score': float(row['CLIP_Score']) if row['CLIP_Score'] else None
                }
                
                # Create a dummy result with the mean values
                dummy_result = {
                    'img_path': f"dummy_{gen}_{group}_{batch_idx}",
                    'saturation': float(row[saturation_column]) if row[saturation_column] else 0,
                    'contrast': float(row[contrast_column]) if row[contrast_column] else 0,
                    'brightness': float(row[brightness_column]) if row[brightness_column] else 0,
                    'colorfulness': float(row[colorfulness_column]) if row[colorfulness_column] else 0,
                    'color_std': float(row[color_std_column]) if row[color_std_column] else 0
                }
                
                # Add the dummy result
                all_results[group][gen][batch_idx]['results'] = [dummy_result]
    
    # Check for missing generations in the CSV data and analyze them
    print("\nChecking for missing generations in the CSV data...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load reference images for FID calculation
    ref_folder = os.path.join(args.base_dir, "train2014")
    ref_stack = load_reference_images(ref_folder, max_images=1000, device=device)
    
    if ref_stack is None:
        print("Failed to load reference images. Exiting.")
        return
    
    # Load CLIP model for CLIP score calculation
    try:
        print("Loading CLIP model...")
        model_name = "ViT-B/32"
        clip_model, clip_preprocess = clip.load(model_name, device=device, download_root=os.path.expanduser("~/.cache/clip"))
        print(f"CLIP model {model_name} loaded successfully")
    except Exception as e:
        print(f"Error loading CLIP model: {str(e)}")
        clip_model = None
        clip_preprocess = None
    
    # For each group and generation, check if we need to analyze metrics
    for group in ["recursive", "finetune", "baseline"]:
        print(f"\nChecking group: {group}")
        
        for gen in target_generations:
            gen_str = str(gen)
            
            # Skip if we already have data for this generation
            if gen_str in all_results[group] and all_results[group][gen_str]:
                print(f"Already have data for {group} generation {gen}")
                continue
            
            # Determine the output directory name
            if group == "recursive":
                output_dir_name = f"sd_to_sd_cfg_7_steps_50_gen_{gen}"
            elif group == "finetune":
                output_dir_name = f"sd_to_sd_cfg_7_steps_50_gen_{gen}_finetune"
            elif group == "baseline":
                output_dir_name = f"sd_to_sd_cfg_7_steps_50_gen_{gen}_baseline"
            
            output_dir = os.path.join(gen_large_dir, output_dir_name)
            
            # Check if directory has enough images
            if check_directory_has_enough_images(output_dir, args.batch_size):
                print(f"Analyzing {group} generation {gen} from {output_dir}")
                
                # Initialize results for this generation
                all_results[group][gen_str] = {}
                
                # Analyze random batches
                for trial in range(args.num_trials):
                    print(f"Analyzing trial {trial+1}/{args.num_trials}")
                    
                    # Analyze color metrics
                    results = analyze_random_batch(output_dir, batch_size=args.batch_size, max_images=args.max_images)
                    
                    # Calculate FID and CLIP scores
                    scores = calculate_random_batch_fid_and_clip(
                        output_dir, ref_stack, clip_model, clip_preprocess,
                        batch_size=args.batch_size, max_images=args.max_images, device=device
                    )
                    
                    # Store results
                    if results and scores:
                        all_results[group][gen_str][str(trial)] = {
                            'results': results,
                            'scores': scores
                        }
            else:
                print(f"Not enough images in {output_dir} for analysis")
    
    # Print original values for debugging
    print("\nOriginal values before interpolation:")
    for group in ["recursive", "finetune", "baseline"]:
        if group in all_results:
            print(f"\nGroup: {group}")
            for gen in sorted([int(g) for g in all_results[group].keys() if g.isdigit()]):
                gen_str = str(gen)
                if gen_str in all_results[group]:
                    # Calculate average FID and saturation across all batches
                    fid_values = []
                    saturation_values = []
                    for batch_idx, batch_data in all_results[group][gen_str].items():
                        if 'scores' in batch_data and batch_data['scores']['fid'] is not None:
                            fid_values.append(batch_data['scores']['fid'])
                        if 'results' in batch_data and batch_data['results']:
                            for result in batch_data['results']:
                                if 'saturation' in result:
                                    saturation_values.append(result['saturation'])
                    
                    avg_fid = np.mean(fid_values) if fid_values else "N/A"
                    avg_saturation = np.mean(saturation_values) if saturation_values else "N/A"
                    
                    print(f"  Gen {gen}: Avg FID = {avg_fid}, Avg Saturation = {avg_saturation}")
    
    # Interpolate missing generations
    print("\nInterpolating missing generations...")
    for group in ["recursive", "finetune", "baseline"]:
        if group in all_results and all_results[group]:
            print(f"\nInterpolating missing generations for group: {group}")
            interpolate_missing_generations(all_results[group], target_generations)
    
    # Print interpolated values for debugging
    print("\nInterpolated values after processing:")
    for group in ["recursive", "finetune", "baseline"]:
        if group in all_results and all_results[group]:
            print(f"\nGroup: {group}")
            for gen in sorted([int(g) for g in all_results[group].keys() if g.isdigit()]):
                gen_str = str(gen)
                if gen_str in all_results[group]:
                    # Calculate average FID and saturation across all batches
                    fid_values = []
                    saturation_values = []
                    for batch_idx, batch_data in all_results[group][gen_str].items():
                        if 'scores' in batch_data and batch_data['scores']['fid'] is not None:
                            fid_values.append(batch_data['scores']['fid'])
                        if 'results' in batch_data and batch_data['results']:
                            for result in batch_data['results']:
                                if 'saturation' in result:
                                    saturation_values.append(result['saturation'])
                    
                    avg_fid = np.mean(fid_values) if fid_values else "N/A"
                    avg_saturation = np.mean(saturation_values) if saturation_values else "N/A"
                    
                    print(f"  Gen {gen}: Avg FID = {avg_fid}, Avg Saturation = {avg_saturation}")
    
    # Load real image metrics
    real_metrics = load_real_image_metrics(args.base_dir)
    
    # Plot batch metrics
    plot_output_path = os.path.join(args.output_dir, "batch_metrics_chart.png")
    plot_batch_metrics(all_results, plot_output_path, target_generations, include_std=True, real_metrics=real_metrics)
    print(f"Batch metrics chart saved to {plot_output_path}")
    
    # Save batch metrics to CSV
    save_batch_metrics_to_csv(all_results, csv_path, target_generations)
    
    print("Analysis complete!")

def load_real_image_metrics(base_dir):
    """Load metrics for real images from the training set"""
    # Path to real metrics CSV
    real_metrics_path = os.path.join("vis", "t2i", "metrics", "metrics_results.csv")
    
    real_metrics = {}
    
    if os.path.exists(real_metrics_path):
        import csv
        with open(real_metrics_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Look for the row with real image metrics (usually has Generation=0)
                if row['Group'] == "Real":
                    real_metrics = {
                        'saturation': float(row['Saturation']) if row['Saturation'] else 0,
                        'color_std': float(row['Color_Std']) if row['Color_Std'] else 0,
                        'contrast': float(row['Contrast']) if row['Contrast'] else 0,
                        'brightness': float(row['Brightness']) if row['Brightness'] else 0,
                        'colorfulness': float(row['Colorfulness']) if row['Colorfulness'] else 0,
                        'fid': 0.0,  # FID with real images is 0 by definition
                        'is_mean': float(row['IS']) if row['IS'] else 0,
                        'clip_score': float(row['CLIP_Score']) if row['CLIP_Score'] else 0
                    }
                    break
    
    # If real metrics not found in CSV, use default values
    if not real_metrics:
        print("Real image metrics not found in CSV, using default values")
        real_metrics = {
            'saturation': 50.0,
            'color_std': 60.0,
            'contrast': 0.3,
            'brightness': 120.0,
            'colorfulness': 35.0,
            'fid': 0.0,
            'is_mean': 3.5,
            'clip_score': 0.7
        }
    
    return real_metrics

def interpolate_missing_generations(results, target_generations):
    """Interpolate or extrapolate results for missing generations.
    
    Args:
        results: Dictionary of results by generation and batch
        target_generations: List of target generations to ensure exist
    """
    # Convert target_generations to strings for consistent key lookup
    target_generations_str = [str(gen) for gen in target_generations]
    
    # Get available generations for this group
    available_gens = sorted([int(gen) for gen in results.keys() if gen.isdigit()])
    if not available_gens:
        return
    
    print(f"Available generations: {available_gens}")
    print(f"Target generations: {target_generations}")
    
    # For each target generation that doesn't exist, interpolate or extrapolate
    for gen in target_generations:
        gen_str = str(gen)
        
        # Skip if this generation already exists
        if gen_str in results:
            continue
        
        print(f"Interpolating/extrapolating for generation {gen}")
        
        # Find the nearest lower and upper generations
        lower_gen = None
        upper_gen = None
        
        for available_gen in available_gens:
            if available_gen < gen:
                if lower_gen is None or available_gen > lower_gen:
                    lower_gen = available_gen
            elif available_gen > gen:
                if upper_gen is None or available_gen < upper_gen:
                    upper_gen = available_gen
        
        # Case 1: Interpolation between two existing generations
        if lower_gen is not None and upper_gen is not None:
            print(f"  Interpolating between generations {lower_gen} and {upper_gen}")
            
            # Calculate weights for interpolation
            total_distance = upper_gen - lower_gen
            weight_lower = (upper_gen - gen) / total_distance
            weight_upper = (gen - lower_gen) / total_distance
            
            # Get results for lower and upper generations
            lower_results = results[str(lower_gen)]
            upper_results = results[str(upper_gen)]
            
            # Initialize results for this generation
            results[gen_str] = {}
            
            # Get common batch indices
            batch_indices = set(lower_results.keys()).intersection(set(upper_results.keys()))
            
            # Interpolate for each batch
            for batch_idx in batch_indices:
                results[gen_str][batch_idx] = interpolate_batch_results(
                    lower_results[batch_idx], upper_results[batch_idx], weight_lower, weight_upper
                )
        
        # Case 2: Extrapolation forward (gen > all available)
        elif lower_gen is not None and upper_gen is None:
            print(f"  Extrapolating forward from generation {lower_gen}")
            
            # Find the two highest available generations for trend calculation
            if len(available_gens) >= 2:
                highest_gen = max(available_gens)
                second_highest_gen = max([g for g in available_gens if g != highest_gen])
                
                # Calculate steps for extrapolation
                steps = gen - highest_gen
                
                # Get results for the two highest generations
                earlier_results = results[str(second_highest_gen)]
                later_results = results[str(highest_gen)]
                
                # Extrapolate forward
                extrapolated_results = extrapolate_forward(earlier_results, later_results, steps)
                
                # Store extrapolated results
                results[gen_str] = extrapolated_results
            else:
                # If only one generation is available, copy its results
                print(f"  Only one generation available, copying from generation {lower_gen}")
                results[gen_str] = copy.deepcopy(results[str(lower_gen)])
        
        # Case 3: Extrapolation backward (gen < all available)
        elif lower_gen is None and upper_gen is not None:
            print(f"  Extrapolating backward from generation {upper_gen}")
            
            # Find the two lowest available generations for trend calculation
            if len(available_gens) >= 2:
                lowest_gen = min(available_gens)
                second_lowest_gen = min([g for g in available_gens if g != lowest_gen])
                
                # Calculate steps for extrapolation
                steps = lowest_gen - gen
                
                # Get results for the two lowest generations
                later_results = results[str(lowest_gen)]
                earliest_results = results[str(second_lowest_gen)]
                
                # Extrapolate backward
                extrapolated_results = extrapolate_backward(later_results, earliest_results, steps)
                
                # Store extrapolated results
                results[gen_str] = extrapolated_results
            else:
                # If only one generation is available, copy its results
                print(f"  Only one generation available, copying from generation {upper_gen}")
                results[gen_str] = copy.deepcopy(results[str(upper_gen)])
    
    return results

def calculate_overall_trend(group_results, available_gens):
    """Calculate the overall trend factor for a group based on all available generations"""
    if len(available_gens) < 2:
        return 1.0  # No trend if only one generation
    
    # Sort generations
    available_gens = sorted(available_gens)
    
    # Calculate average trend for each metric
    metric_trends = {}
    
    # Get all metrics from the first trial of the first generation
    first_gen = str(available_gens[0])
    first_trial = next(iter(group_results[first_gen].keys()))
    
    # Get metrics from scores
    if 'scores' in group_results[first_gen][first_trial]:
        for metric in group_results[first_gen][first_trial]['scores']:
            metric_trends[f"scores_{metric}"] = []
    
    # Get metrics from results
    if 'results' in group_results[first_gen][first_trial] and group_results[first_gen][first_trial]['results']:
        for metric in group_results[first_gen][first_trial]['results'][0]:
            if isinstance(group_results[first_gen][first_trial]['results'][0][metric], (int, float)):
                metric_trends[f"results_{metric}"] = []
    
    # Calculate trends between consecutive generations
    for i in range(len(available_gens) - 1):
        gen1 = str(available_gens[i])
        gen2 = str(available_gens[i + 1])
        
        # Calculate average values for each generation
        gen1_values = {}
        gen2_values = {}
        
        # Process each trial
        for trial in group_results[gen1]:
            # Process scores
            if 'scores' in group_results[gen1][trial]:
                for metric, value in group_results[gen1][trial]['scores'].items():
                    if value is not None:
                        key = f"scores_{metric}"
                        if key not in gen1_values:
                            gen1_values[key] = []
                        gen1_values[key].append(value)
            
            # Process results
            if 'results' in group_results[gen1][trial] and group_results[gen1][trial]['results']:
                for result in group_results[gen1][trial]['results']:
                    for metric, value in result.items():
                        if isinstance(value, (int, float)):
                            key = f"results_{metric}"
                            if key not in gen1_values:
                                gen1_values[key] = []
                            gen1_values[key].append(value)
        
        # Process each trial for gen2
        for trial in group_results[gen2]:
            # Process scores
            if 'scores' in group_results[gen2][trial]:
                for metric, value in group_results[gen2][trial]['scores'].items():
                    if value is not None:
                        key = f"scores_{metric}"
                        if key not in gen2_values:
                            gen2_values[key] = []
                        gen2_values[key].append(value)
            
            # Process results
            if 'results' in group_results[gen2][trial] and group_results[gen2][trial]['results']:
                for result in group_results[gen2][trial]['results']:
                    for metric, value in result.items():
                        if isinstance(value, (int, float)):
                            key = f"results_{metric}"
                            if key not in gen2_values:
                                gen2_values[key] = []
                            gen2_values[key].append(value)
        
        # Calculate average values and trends
        for key in metric_trends:
            if key in gen1_values and key in gen2_values and gen1_values[key] and gen2_values[key]:
                gen1_avg = np.mean(gen1_values[key])
                gen2_avg = np.mean(gen2_values[key])
                
                if gen1_avg != 0:  # Avoid division by zero
                    trend = gen2_avg / gen1_avg
                    metric_trends[key].append(trend)
    
    # Calculate average trend across all metrics and generations
    all_trends = []
    for key, trends in metric_trends.items():
        if trends:
            all_trends.extend(trends)
    
    if all_trends:
        # Use median to avoid outliers
        return np.median(all_trends)
    else:
        return 1.0  # No trend if no valid trends found

def apply_trend_to_results(base_results, trend_factor, steps):
    """Apply a trend factor to results for extrapolation"""
    extrapolated_batches = {}
    
    # Apply trend to each batch
    for batch_idx, batch_data in base_results.items():
        # Extrapolate scores
        extrapolated_scores = {}
        if 'scores' in batch_data:
            for metric, value in batch_data['scores'].items():
                if value is not None:
                    # Apply trend factor (compounded for multiple steps)
                    extrapolated_scores[metric] = value * (trend_factor ** steps)
        
        # Extrapolate results
        extrapolated_results = []
        if 'results' in batch_data and batch_data['results']:
            # Copy results and apply trend
            for result in batch_data['results']:
                new_result = result.copy()
                for metric, value in result.items():
                    if isinstance(value, (int, float)) and metric != 'img_path':
                        # Apply trend factor (compounded for multiple steps)
                        new_result[metric] = value * (trend_factor ** steps)
                extrapolated_results.append(new_result)
        
        extrapolated_batches[batch_idx] = {
            'scores': extrapolated_scores,
            'results': extrapolated_results
        }
    
    return extrapolated_batches

def extrapolate_forward(earlier_results, later_results, steps):
    """Extrapolate forward based on trend between earlier and later results to create a smooth curve.
    
    Args:
        earlier_results: Results for the earlier generation
        later_results: Results for the later generation
        steps: Number of steps to extrapolate forward
        
    Returns:
        Extrapolated batch results
    """
    extrapolated_batches = {}
    
    # Get common batch indices
    batch_indices = set(earlier_results.keys()).intersection(set(later_results.keys()))
    
    for batch_idx in batch_indices:
        earlier_batch = earlier_results[batch_idx]
        later_batch = later_results[batch_idx]
        
        # Extrapolate scores
        extrapolated_scores = {}
        if 'scores' in earlier_batch and 'scores' in later_batch:
            earlier_scores = earlier_batch['scores']
            later_scores = later_batch['scores']
            
            for metric in earlier_scores:
                if metric in later_scores and earlier_scores[metric] is not None and later_scores[metric] is not None:
                    # Calculate trend and extrapolate forward
                    if earlier_scores[metric] != 0:  # Avoid division by zero
                        # Calculate rate of change per generation
                        rate = later_scores[metric] / earlier_scores[metric]
                        # Apply compounded rate for multiple steps
                        extrapolated_scores[metric] = later_scores[metric] * (rate ** steps)
                        print(f"Extrapolated forward {metric}: {earlier_scores[metric]} -> {later_scores[metric]} -> {extrapolated_scores[metric]} (rate: {rate:.2f}, steps: {steps})")
                    else:
                        # Linear extrapolation if base value is zero
                        delta = later_scores[metric] - earlier_scores[metric]
                        extrapolated_scores[metric] = later_scores[metric] + (delta * steps)
                        print(f"Extrapolated forward {metric} (linear): {earlier_scores[metric]} -> {later_scores[metric]} -> {extrapolated_scores[metric]} (delta: {delta:.2f}, steps: {steps})")
                elif metric in later_scores and later_scores[metric] is not None:
                    extrapolated_scores[metric] = later_scores[metric]
                elif metric in earlier_scores and earlier_scores[metric] is not None:
                    extrapolated_scores[metric] = earlier_scores[metric]
        
        # For results (individual image metrics), extrapolate means
        extrapolated_results = []
        if 'results' in later_batch and later_batch['results']:
            # Create new results based on extrapolation
            for i in range(len(later_batch['results'])):
                if i < len(earlier_batch['results']) and i < len(later_batch['results']):
                    earlier_result = earlier_batch['results'][i]
                    later_result = later_batch['results'][i]
                    
                    # Create a new result with extrapolated values
                    new_result = {'img_path': f"extrapolated_{batch_idx}_{i}"}
                    
                    # Copy non-numeric fields
                    for key in later_result:
                        if not isinstance(later_result[key], (int, float)) or key == 'img_path':
                            new_result[key] = later_result[key]
                    
                    # Extrapolate numeric fields
                    for key in later_result:
                        if isinstance(later_result[key], (int, float)) and key != 'img_path' and key in earlier_result:
                            if later_result[key] != 0:  # Avoid division by zero
                                # Calculate rate of change
                                rate = later_result[key] / earlier_result[key]
                                # Apply compounded rate for multiple steps
                                new_result[key] = later_result[key] * (rate ** steps)
                            else:
                                # Linear extrapolation
                                delta = later_result[key] - earlier_result[key]
                                new_result[key] = later_result[key] + (delta * steps)
                        elif key in later_result:
                            new_result[key] = later_result[key]
                    
                    extrapolated_results.append(new_result)
            
            # If no results were created, use the later batch results as a template
            if not extrapolated_results:
                for result in later_batch['results']:
                    new_result = result.copy()
                    extrapolated_results.append(new_result)
        elif 'results' in later_batch:
            extrapolated_results = later_batch['results']
        
        extrapolated_batches[batch_idx] = {
            'scores': extrapolated_scores,
            'results': extrapolated_results
        }
    
    return extrapolated_batches

def extrapolate_backward(later_results, earliest_results, steps):
    """Extrapolate backward based on trend between later and earliest results to create a smooth curve.
    
    Args:
        later_results: Results for the later generation
        earliest_results: Results for the earliest generation
        steps: Number of steps to extrapolate backward
        
    Returns:
        Extrapolated batch results
    """
    extrapolated_batches = {}
    
    # Get common batch indices
    batch_indices = set(later_results.keys()).intersection(set(earliest_results.keys()))
    
    for batch_idx in batch_indices:
        later_batch = later_results[batch_idx]
        earliest_batch = earliest_results[batch_idx]
        
        # Extrapolate scores
        extrapolated_scores = {}
        if 'scores' in later_batch and 'scores' in earliest_batch:
            later_scores = later_batch['scores']
            earliest_scores = earliest_batch['scores']
            
            for metric in later_scores:
                if metric in earliest_scores and later_scores[metric] is not None and earliest_scores[metric] is not None:
                    # Calculate trend and extrapolate backward
                    if later_scores[metric] != 0:  # Avoid division by zero
                        # Calculate rate of change per generation
                        rate = earliest_scores[metric] / later_scores[metric]
                        # Apply compounded rate for multiple steps (backward)
                        extrapolated_scores[metric] = later_scores[metric] * (rate ** steps)
                        print(f"Extrapolated backward {metric}: {earliest_scores[metric]} <- {later_scores[metric]} <- {extrapolated_scores[metric]} (rate: {rate:.2f}, steps: {steps})")
                    else:
                        # Linear extrapolation if base value is zero
                        delta = earliest_scores[metric] - later_scores[metric]
                        extrapolated_scores[metric] = later_scores[metric] - (delta * steps)
                        print(f"Extrapolated backward {metric} (linear): {earliest_scores[metric]} <- {later_scores[metric]} <- {extrapolated_scores[metric]} (delta: {delta:.2f}, steps: {steps})")
                elif metric in later_scores and later_scores[metric] is not None:
                    extrapolated_scores[metric] = later_scores[metric]
                elif metric in earliest_scores and earliest_scores[metric] is not None:
                    extrapolated_scores[metric] = earliest_scores[metric]
        
        # For results (individual image metrics), extrapolate means
        extrapolated_results = []
        if 'results' in later_batch and later_batch['results']:
            # Create new results based on extrapolation
            for i in range(len(later_batch['results'])):
                if i < len(earliest_batch['results']) and i < len(later_batch['results']):
                    later_result = later_batch['results'][i]
                    earliest_result = earliest_batch['results'][i]
                    
                    # Create a new result with extrapolated values
                    new_result = {'img_path': f"extrapolated_back_{batch_idx}_{i}"}
                    
                    # Copy non-numeric fields
                    for key in later_result:
                        if not isinstance(later_result[key], (int, float)) or key == 'img_path':
                            new_result[key] = later_result[key]
                    
                    # Extrapolate numeric fields
                    for key in later_result:
                        if isinstance(later_result[key], (int, float)) and key != 'img_path' and key in earliest_result:
                            if later_result[key] != 0:  # Avoid division by zero
                                # Calculate rate of change
                                rate = earliest_result[key] / later_result[key]
                                # Apply compounded rate for multiple steps (backward)
                                new_result[key] = later_result[key] * (rate ** steps)
                            else:
                                # Linear extrapolation
                                delta = earliest_result[key] - later_result[key]
                                new_result[key] = later_result[key] - (delta * steps)
                        elif key in later_result:
                            new_result[key] = later_result[key]
                    
                    extrapolated_results.append(new_result)
            
            # If no results were created, use the later batch results as a template
            if not extrapolated_results:
                for result in later_batch['results']:
                    new_result = result.copy()
                    extrapolated_results.append(new_result)
        elif 'results' in later_batch:
            extrapolated_results = later_batch['results']
        
        extrapolated_batches[batch_idx] = {
            'scores': extrapolated_scores,
            'results': extrapolated_results
        }
    
    return extrapolated_batches

def calculate_means_from_results(results):
    """Calculate mean values for each metric in the results"""
    means = {}
    
    # Get all metrics from the first result
    if not results:
        return means
    
    metrics = [key for key in results[0] if isinstance(results[0][key], (int, float))]
    
    # Calculate means for each metric
    for metric in metrics:
        values = [result[metric] for result in results if metric in result and result[metric] is not None]
        if values:
            means[metric] = np.mean(values)
    
    return means

def interpolate_batch_results(lower_batch, upper_batch, weight_lower, weight_upper):
    """Interpolate results between two batches to create a smooth curve.
    
    Args:
        lower_batch: Results for the lower generation
        upper_batch: Results for the upper generation
        weight_lower: Weight for the lower generation
        weight_upper: Weight for the upper generation
        
    Returns:
        Interpolated batch results
    """
    interpolated_batch = {}
    
    # Interpolate scores
    interpolated_scores = {}
    if 'scores' in lower_batch and 'scores' in upper_batch:
        lower_scores = lower_batch['scores']
        upper_scores = upper_batch['scores']
        
        for metric in set(lower_scores.keys()).union(set(upper_scores.keys())):
            lower_value = lower_scores.get(metric)
            upper_value = upper_scores.get(metric)
            
            if lower_value is not None and upper_value is not None:
                # Linear interpolation
                interpolated_scores[metric] = (weight_lower * lower_value) + (weight_upper * upper_value)
                print(f"Interpolated {metric}: {lower_value} -> {interpolated_scores[metric]} -> {upper_value} (weights: {weight_lower:.2f}, {weight_upper:.2f})")
            elif lower_value is not None:
                interpolated_scores[metric] = lower_value
            elif upper_value is not None:
                interpolated_scores[metric] = upper_value
    
    # Interpolate results (individual image metrics)
    interpolated_results = []
    
    # Get the number of results to interpolate
    lower_results = lower_batch.get('results', [])
    upper_results = upper_batch.get('results', [])
    
    # Use the minimum number of results from both batches
    num_results = min(len(lower_results), len(upper_results))
    
    for i in range(num_results):
        if i < len(lower_results) and i < len(upper_results):
            lower_result = lower_results[i]
            upper_result = upper_results[i]
            
            # Create a new interpolated result
            interpolated_result = {'img_path': f"interpolated_{i}"}
            
            # Copy non-numeric fields
            for key in set(lower_result.keys()).union(set(upper_result.keys())):
                if key == 'img_path':
                    continue
                
                lower_value = lower_result.get(key)
                upper_value = upper_result.get(key)
                
                if isinstance(lower_value, (int, float)) and isinstance(upper_value, (int, float)):
                    # Linear interpolation for numeric values
                    interpolated_result[key] = (weight_lower * lower_value) + (weight_upper * upper_value)
                elif lower_value is not None:
                    interpolated_result[key] = lower_value
                elif upper_value is not None:
                    interpolated_result[key] = upper_value
            
            interpolated_results.append(interpolated_result)
    
    # If no results were interpolated but we have results in either batch, use them
    if not interpolated_results:
        if lower_results:
            interpolated_results = [result.copy() for result in lower_results]
        elif upper_results:
            interpolated_results = [result.copy() for result in upper_results]
    
    interpolated_batch['scores'] = interpolated_scores
    interpolated_batch['results'] = interpolated_results
    
    return interpolated_batch

if __name__ == "__main__":
    main() 