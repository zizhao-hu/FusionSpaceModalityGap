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
from matplotlib import gridspec

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
        if model_path != "original" and os.path.exists(os.path.join(model_path, "unet")):
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

def analyze_random_batch(image_dir, batch_size=100, max_images=200, gen=None):
    """Analyze metrics for a random batch of images"""
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    
    # Limit to max_images
    if len(image_files) > max_images:
        image_files = image_files[:max_images]
    
    if len(image_files) < batch_size:
        print(f"Not enough images in {image_dir}. Found {len(image_files)}, need {batch_size}")
        return None
    
    # Use a fixed seed for generation 0 and 10 to ensure consistent results
    if gen == 0 or gen == 10:
        print(f"Using fixed random seed for generation {gen}")
        # Note: For generation 0, the seed is set by the caller to ensure consistency across models
        # For generation 10, we use a fixed seed
        if gen == 10:
            np.random.seed(42)
    else:
        # Use different seed each time for other generations
        np.random.seed(None)
    
    # Randomly select batch_size images
    selected_files = np.random.choice(image_files, size=batch_size, replace=False)
    
    # Analyze color distribution
    results = []
    for img_path in tqdm(selected_files, desc=f"Analyzing random batch"):
        color_data = analyze_color_distribution(img_path)
        if color_data:
            results.append(color_data)
    
    return results

def calculate_random_batch_fid_and_clip(image_dir, ref_stack, clip_model=None, clip_preprocess=None, 
                                      batch_size=100, max_images=200, device="cuda", gen=None):
    """Calculate FID and CLIP scores for a random batch of images"""
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    
    # Limit to max_images
    if len(image_files) > max_images:
        image_files = image_files[:max_images]
    
    if len(image_files) < batch_size:
        print(f"Not enough images in {image_dir}. Found {len(image_files)}, need {batch_size}")
        return None
    
    # Use a fixed seed for generation 0 and 10 to ensure consistent results
    if gen == 0 or gen == 10:
        print(f"Using fixed random seed for FID/CLIP calculation for generation {gen}")
        # Note: For generation 0, the seed is set by the caller to ensure consistency across models
        # For generation 10, we use a fixed seed
        if gen == 10:
            np.random.seed(42)
    else:
        # Use different seed each time for other generations
        np.random.seed(None)
    
    # Randomly select batch_size images
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
    scores = {'fid': None, 'is_mean': None, 'is_std': None, 'clip_score': None, 'rmg': None, 'l2m': None, 'clip_variance': None}
    
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
    
    # Calculate CLIP score and additional metrics if model is available
    if clip_model is not None and clip_preprocess is not None:
        try:
            # Simple CLIP score calculation
            clip_scores = []
            
            # Process in smaller batches
            clip_batch_size = 16
            
            # Store image and text features for additional metrics
            all_image_features = []
            all_text_features = []
            
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
                
                # Store features for additional metrics
                all_image_features.append(image_features.cpu().numpy())
                all_text_features.append(text_features.cpu().numpy())
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
                
                # Calculate similarity
                similarities = (image_features @ text_features.T).diag()
                similarities = (similarities + 1) / 2  # Convert from -1,1 to 0,1
                clip_scores.extend(similarities.cpu().numpy())
            
            scores['clip_score'] = float(np.mean(clip_scores))
            
            # Calculate additional metrics from plot_gap.py
            all_image_features = np.vstack(all_image_features)
            all_text_features = np.vstack(all_text_features)
            
            # Calculate CLIP variance (embedding variance)
            scores['clip_variance'] = float(np.mean(np.var(all_image_features, axis=0)))
            
            # Calculate L2M (L2 distance between mean embeddings)
            scores['l2m'] = float(np.linalg.norm(np.mean(all_text_features, axis=0) - np.mean(all_image_features, axis=0)))
            
            # Calculate RMG (Relative Multimodal Gap)
            # Implementation based on plot_gap.py's rmg_cosine_dissimilarity
            def cosine_dissim_rowwise(A, B):
                numerator = np.einsum('ij,ij->i', A, B)
                normA = np.linalg.norm(A, axis=1)
                normB = np.linalg.norm(B, axis=1)
                cos_sim = numerator / (normA * normB)
                return 1.0 - cos_sim  # dissimilarity
            
            N = all_image_features.shape[0]
            row_dissim_xy = cosine_dissim_rowwise(all_text_features, all_image_features)
            numerator = np.mean(row_dissim_xy)
            
            def sum_pairwise_cos_dissim(M):
                dot_mat = M @ M.T
                norms = np.linalg.norm(M, axis=1)
                norm_mat = np.outer(norms, norms)
                cos_sim_mat = dot_mat / norm_mat
                cos_dissim_mat = 1.0 - cos_sim_mat
                return np.sum(cos_dissim_mat)
            
            sum_dxx = sum_pairwise_cos_dissim(all_text_features)
            sum_dyy = sum_pairwise_cos_dissim(all_image_features)
            denom_part1 = (1.0 / (2.0 * N * (N - 1))) * (sum_dxx + sum_dyy)
            denom_part2 = numerator
            denominator = denom_part1 + denom_part2
            rmg_value = numerator / denominator
            
            scores['rmg'] = float(rmg_value)
            
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error calculating CLIP metrics: {e}")
    
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

def plot_batch_metrics(all_results, output_path, target_generations=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], include_std=True, real_metrics=None):
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
    
    # Define all the metrics to plot (reordered to put FID, IS, CLIP score last)
    all_metrics = [
        # Color metrics 
        {'name': 'Saturation', 'key': 'saturation', 'source': 'results'},
        {'name': 'Contrast', 'key': 'contrast', 'source': 'results'},
        {'name': 'Brightness', 'key': 'brightness', 'source': 'results'},
        {'name': 'Color Std', 'key': 'color_std', 'source': 'results'},
        {'name': 'RMG', 'key': 'rmg', 'source': 'scores'},
        
        # Second row
        {'name': 'L2M', 'key': 'l2m', 'source': 'scores', 'scale_factor': 0.1, 'offset': -0.4},  # Scale down L2M and subtract offset (moved up by 0.1)
        {'name': 'CLIP Variance', 'key': 'clip_variance', 'source': 'scores'},
        {'name': 'FID', 'key': 'fid', 'source': 'scores'},
        {'name': 'IS', 'key': 'is_mean', 'source': 'scores'},
        {'name': 'CLIP Score', 'key': 'clip_score', 'source': 'scores'}
    ]
    
    # Fixed layout: 2 rows, 5 columns
    n_rows = 2
    n_cols = 5
    
    # Define the specific generations to show on x-axis ticks
    display_generations = [0, 4, 9]  # Changed from [0, 5, 10] to [0, 4, 9]
    
    # Increase font sizes globally
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 22,
        'axes.labelsize': 20,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 14,  # Smaller legend font size
    })
    
    # Create a blank figure with a generous size
    fig_size = 4  # Size of each subplot in inches
    spacing = 1.2  # Spacing factor between subplots
    fig_width = fig_size * n_cols * spacing + 2
    fig_height = fig_size * n_rows * spacing + 2
    
    # Create the figure
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Calculate consistent Gen 0 values across all groups
    # This ensures all groups use the same Gen 0 reference
    consistent_gen0_values = {}
    for metric in all_metrics:
        metric_key = metric['key']
        source = metric['source']
        scale_factor = metric.get('scale_factor', 1.0)
        offset = metric.get('offset', 0.0)
        
        # Gather all Gen 0 values for this metric across all groups
        all_gen0_values = []
        for group in groups:
            group_key = group["key"]
            if group_key in all_results and "0" in all_results[group_key]:
                for trial_idx, trial_data in all_results[group_key]["0"].items():
                    if source == 'scores':
                        # Get from scores dictionary
                        if 'scores' in trial_data and metric_key in trial_data['scores'] and trial_data['scores'][metric_key] is not None:
                            # Apply scale factor and offset if needed
                            processed_value = trial_data['scores'][metric_key] * scale_factor
                            if metric_key == 'l2m':  # Apply offset only to L2M
                                processed_value += offset
                            all_gen0_values.append(processed_value)
                    else:
                        # Calculate from results dictionary
                        if 'results' in trial_data:
                            # Calculate mean for this trial
                            img_values = []
                            for img_data in trial_data['results']:
                                if metric_key in img_data and img_data[metric_key] is not None:
                                    img_values.append(img_data[metric_key])
                            if img_values:
                                all_gen0_values.append(np.mean(img_values))
        
        # Calculate consistent Gen 0 value if we have data
        if all_gen0_values:
            consistent_gen0_values[metric_key] = np.mean(all_gen0_values)
    
    # Plot each metric in its own square subplot
    for i, metric in enumerate(all_metrics):
        # Calculate row and column for this metric
        row = i // n_cols
        col = i % n_cols
        
        # Create a square subplot at a specific position
        # The position is defined by [left, bottom, width, height] in figure coordinates
        ax_width = fig_size / fig_width
        ax_height = fig_size / fig_height
        left = (col * fig_size * spacing + 1) / fig_width
        bottom = 1 - ((row + 1) * fig_size * spacing - 0.5) / fig_height
        
        ax = fig.add_axes([left, bottom, ax_width, ax_height])
        ax.set_facecolor('white')
        
        metric_key = metric['key']
        source = metric['source']
        scale_factor = metric.get('scale_factor', 1.0)
        offset = metric.get('offset', 0.0)
        
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
            
            # Get all available generations for this group
            available_gens = sorted([int(gen) for gen in all_results[group_key].keys() if gen.isdigit() and int(gen) <= 9])  # Exclude gen 10
            
            for gen in available_gens:
                gen_str = str(gen)
                
                if gen_str not in all_results[group_key]:
                    continue
                
                # Calculate mean and std across trials
                trial_means = []
                
                for trial_idx, trial_data in all_results[group_key][gen_str].items():
                    if source == 'scores':
                        # Get from scores dictionary
                        if 'scores' in trial_data and metric_key in trial_data['scores'] and trial_data['scores'][metric_key] is not None:
                            # Apply scale factor and offset if needed
                            processed_value = trial_data['scores'][metric_key] * scale_factor
                            if metric_key == 'l2m':  # Apply offset only to L2M
                                processed_value += offset
                            trial_means.append(processed_value)
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
                               linewidth=3, markersize=10, color=group_color, label=group_name)
                lines.append(line)
                labels.append(group_name)
                
                # Plot standard deviation as shaded area if requested
                if include_std:
                    ax.fill_between(x_values, 
                                   [y - s for y, s in zip(y_means, y_stds)],
                                   [y + s for y, s in zip(y_means, y_stds)],
                                   color=group_color, alpha=0.2)
        
        # Add consistent Gen 0 reference line
        if metric_key in consistent_gen0_values:
            gen0_value = consistent_gen0_values[metric_key]
            # Plot the consistent Gen 0 value for all groups with the same style
            ax.axhline(y=gen0_value, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='GEN 0')
            
            # Add to legend
            lines.append(plt.Line2D([0], [0], color='gray', linestyle='--', linewidth=2, alpha=0.7))
            labels.append('GEN 0')
        
        # Add real image reference line using hardcoded values
        # Skip adding reference line for FID
        if metric_key in hardcoded_real_metrics and metric_key != 'fid':
            real_value = hardcoded_real_metrics[metric_key]
            
            # Apply scale factor and offset if needed
            if metric_key == 'l2m':
                real_value = real_value * scale_factor + offset
            
            # Plot the real value as a dashed line
            ax.axhline(y=real_value, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Real Images')
            
            # Add to legend
            lines.append(plt.Line2D([0], [0], color='black', linestyle='--', linewidth=2, alpha=0.5))
            labels.append('Real Images')
        
        # Set x-axis ticks to only show GEN 0, GEN 4, GEN 9
        ax.set_xticks(display_generations)
        ax.set_xticklabels([f"GEN {x}" for x in display_generations], fontsize=18)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Set title
        metric_title = metric['name']
        # For L2M, just show the name without scale or offset info
        ax.set_title(metric_title, fontsize=22, pad=10)
        
        # Increase y-axis tick font size
        ax.tick_params(axis='y', labelsize=18)
        
        # Force square aspect ratio
        ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
        
        # Add legend to the first subplot only (Saturation plot)
        if i == 0:
            ax.legend(lines, labels, loc='lower right', fontsize=14, framealpha=0.7, frameon=True)
    
    # Add more space between subplots
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    # Save figure with sufficient DPI
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Batch metrics chart saved to {output_path}")
    plt.close()

def save_batch_metrics_to_csv(all_results, output_path, target_generations=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
    """Save batch metrics to a CSV file.
    
    Args:
        all_results: Dictionary of results by group, generation, and batch
        output_path: Path to save the CSV file
        target_generations: List of generations to include
    """
    # Define the groups
    groups = ["recursive", "finetune", "baseline"]
    
    with open(output_path, 'w') as f:
        # Write header with new metrics
        f.write("Generation,Group,Batch,Saturation,Saturation_Std,Contrast,Contrast_Std,Brightness,Brightness_Std,Color_Std,Color_Std_Std,FID,IS,CLIP_Score,RMG,L2M,CLIP_Variance\n")
        
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
                        color_std_values = [r.get('color_std', 0) for r in batch_data['results'] if 'color_std' in r]
                        
                        # Calculate means and standard deviations
                        saturation_mean = np.mean(saturation_values) if saturation_values else 0
                        saturation_std = np.std(saturation_values) if saturation_values else 0
                        
                        contrast_mean = np.mean(contrast_values) if contrast_values else 0
                        contrast_std = np.std(contrast_values) if contrast_values else 0
                        
                        brightness_mean = np.mean(brightness_values) if brightness_values else 0
                        brightness_std = np.std(brightness_values) if brightness_values else 0
                        
                        color_std_mean = np.mean(color_std_values) if color_std_values else 0
                        color_std_std = np.std(color_std_values) if color_std_values else 0
                    else:
                        # No results available
                        saturation_mean = saturation_std = 0
                        contrast_mean = contrast_std = 0
                        brightness_mean = brightness_std = 0
                        color_std_mean = color_std_std = 0
                    
                    # Get scores including new metrics
                    fid = is_score = clip_score = rmg = l2m = clip_variance = ""
                    if 'scores' in batch_data:
                        scores = batch_data['scores']
                        fid = scores.get('fid', "")
                        is_score = scores.get('is_mean', "")
                        clip_score = scores.get('clip_score', "")
                        rmg = scores.get('rmg', "")
                        l2m = scores.get('l2m', "")
                        clip_variance = scores.get('clip_variance', "")
                    
                    # Write row with new metrics
                    f.write(f"{gen},{display_group},{batch_idx},{saturation_mean:.2f},{saturation_std:.2f},{contrast_mean:.4f},{contrast_std:.4f},{brightness_mean:.2f},{brightness_std:.2f},{color_std_mean:.2f},{color_std_std:.2f},{fid},{is_score},{clip_score},{rmg},{l2m},{clip_variance}\n")
    
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
    scores = {'fid': None, 'is_mean': None, 'is_std': None, 'clip_score': None, 'rmg': None, 'l2m': None, 'clip_variance': None}
    
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
    
    # Calculate CLIP score and additional metrics if model is available
    if clip_model is not None and clip_preprocess is not None:
        try:
            # Simple CLIP score calculation
            clip_scores = []
            
            # Process in smaller batches
            clip_batch_size = 16
            
            # Store image and text features for additional metrics
            all_image_features = []
            all_text_features = []
            
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
                
                # Store features for additional metrics
                all_image_features.append(image_features.cpu().numpy())
                all_text_features.append(text_features.cpu().numpy())
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
                
                # Calculate similarity
                similarities = (image_features @ text_features.T).diag()
                similarities = (similarities + 1) / 2  # Convert from -1,1 to 0,1
                clip_scores.extend(similarities.cpu().numpy())
            
            scores['clip_score'] = float(np.mean(clip_scores))
            
            # Calculate additional metrics from plot_gap.py
            all_image_features = np.vstack(all_image_features)
            all_text_features = np.vstack(all_text_features)
            
            # Calculate CLIP variance (embedding variance)
            scores['clip_variance'] = float(np.mean(np.var(all_image_features, axis=0)))
            
            # Calculate L2M (L2 distance between mean embeddings)
            scores['l2m'] = float(np.linalg.norm(np.mean(all_text_features, axis=0) - np.mean(all_image_features, axis=0)))
            
            # Calculate RMG (Relative Multimodal Gap)
            # Implementation based on plot_gap.py's rmg_cosine_dissimilarity
            def cosine_dissim_rowwise(A, B):
                numerator = np.einsum('ij,ij->i', A, B)
                normA = np.linalg.norm(A, axis=1)
                normB = np.linalg.norm(B, axis=1)
                cos_sim = numerator / (normA * normB)
                return 1.0 - cos_sim  # dissimilarity
            
            N = all_image_features.shape[0]
            row_dissim_xy = cosine_dissim_rowwise(all_text_features, all_image_features)
            numerator = np.mean(row_dissim_xy)
            
            def sum_pairwise_cos_dissim(M):
                dot_mat = M @ M.T
                norms = np.linalg.norm(M, axis=1)
                norm_mat = np.outer(norms, norms)
                cos_sim_mat = dot_mat / norm_mat
                cos_dissim_mat = 1.0 - cos_sim_mat
                return np.sum(cos_dissim_mat)
            
            sum_dxx = sum_pairwise_cos_dissim(all_text_features)
            sum_dyy = sum_pairwise_cos_dissim(all_image_features)
            denom_part1 = (1.0 / (2.0 * N * (N - 1))) * (sum_dxx + sum_dyy)
            denom_part2 = numerator
            denominator = denom_part1 + denom_part2
            rmg_value = numerator / denominator
            
            scores['rmg'] = float(rmg_value)
            
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error calculating CLIP metrics: {e}")
    
    return scores

def check_directory_has_enough_images(dir_path, min_images=100):
    """Check if a directory has enough images for analysis."""
    if not os.path.exists(dir_path):
        return False
    
    image_files = glob.glob(os.path.join(dir_path, "*.jpg"))
    return len(image_files) >= min_images

def delete_generation_from_csv(csv_path, generation_to_delete):
    """Delete all entries for a specific generation from the CSV file.
    
    Args:
        csv_path: Path to the CSV file
        generation_to_delete: Generation number to delete
    
    Returns:
        bool: True if the file was modified, False otherwise
    """
    if not os.path.exists(csv_path):
        print(f"CSV file {csv_path} does not exist. Nothing to delete.")
        return False
    
    import csv
    
    # Read the existing CSV file
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        
        # Filter out rows with the specified generation
        for row in reader:
            if row['Generation'] != str(generation_to_delete):
                rows.append(row)
    
    # Check if any rows were removed
    if len(rows) == 0 and os.path.getsize(csv_path) > 0:
        print(f"All rows would be deleted. Keeping the header only.")
        # Keep the header only
        with open(csv_path, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
        return True
    
    # Write the filtered data back to the CSV file
    if rows:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Deleted all entries for generation {generation_to_delete} from {csv_path}")
        return True
    
    return False

def ensure_gen0_images_exist(base_dir, available_checkpoints, captions, batch_size=4, max_images=200, device="cuda"):
    """Ensure generation 0 images exist, generating them if necessary.
    
    Args:
        base_dir: Base directory containing the image folders
        available_checkpoints: Dictionary of available model checkpoints
        captions: List of captions for image generation
        batch_size: Batch size for image generation
        max_images: Maximum number of images to generate
        device: Device to use for generation
        
    Returns:
        str: Path to the directory containing generation 0 images
    """
    gen_large_dir = os.path.join(base_dir, "gen_large")
    os.makedirs(gen_large_dir, exist_ok=True)
    
    # Determine the output directory for generation 0 images
    output_dir_name = "sd_to_sd_cfg_7_steps_50_gen_0"
    output_dir = os.path.join(gen_large_dir, output_dir_name)
    
    # Check if directory has enough images
    if not check_directory_has_enough_images(output_dir, batch_size) or len(glob.glob(os.path.join(output_dir, "*.jpg"))) < max_images or os.path.exists(output_dir) == False:
        print(f"Generating images for generation 0 using original Stable Diffusion model in {output_dir}")
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Delete any existing images in the directory
        existing_images = glob.glob(os.path.join(output_dir, "*.jpg"))
        for img_path in existing_images:
            try:
                os.remove(img_path)
                print(f"Deleted existing image: {img_path}")
            except Exception as e:
                print(f"Error deleting image {img_path}: {e}")
        
        # Generate images using the original Stable Diffusion model
        success = generate_images_for_evaluation(
            "original", output_dir, captions, 
            num_images=max_images, batch_size=batch_size, 
            cfg_scale=7.0, steps=50, device=device
        )
        
        if not success:
            print(f"Failed to generate images for generation 0 using original Stable Diffusion model")
            return None
    else:
        print(f"Directory {output_dir} already has enough images for generation 0")
    
    # Verify that we have enough images
    if not check_directory_has_enough_images(output_dir, batch_size):
        print(f"Failed to generate enough images for generation 0 in {output_dir}")
        return None
    
    # Count the number of images
    image_count = len(glob.glob(os.path.join(output_dir, "*.jpg")))
    print(f"Found {image_count} images for generation 0 in {output_dir}")
    
    return output_dir

def main():
    """Main function to analyze image metrics."""
    parser = argparse.ArgumentParser(description='Analyze image metrics for generated images')
    parser.add_argument('--base_dir', type=str, default=os.path.join("data", "coco"), 
                        help='Base directory containing the image folders')
    parser.add_argument('--output_dir', type=str, default=os.path.join("vis", "t2i", "metrics_large"),
                        help='Output directory for charts and CSV')
    parser.add_argument('--target_generations', type=str, default="0,1,2,3,4,5,6,7,8,9,10",
                        help='Comma-separated list of generations to analyze')
    parser.add_argument('--display_generations', type=str, default="0,5,10",
                        help='Comma-separated list of generations to display on x-axis')
    parser.add_argument('--num_trials', type=int, default=10,
                        help='Number of random sampling trials to perform')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Number of images to analyze in each batch')
    parser.add_argument('--max_images', type=int, default=200,
                        help='Maximum number of images to consider per generation')
    parser.add_argument('--delete_gen', type=int, default=None,
                        help='Generation number to delete from CSV before analysis')
    parser.add_argument('--recalculate_gen0', action='store_true', default=True,
                        help='Recalculate generation 0 metrics for all models using the same images')
    parser.add_argument('--vis_only', action='store_true', default=False,
                        help='Visualization only mode: generate plots from existing CSV data without recalculating metrics')
    
    args = parser.parse_args()
    
    # Parse target generations
    target_generations = [int(g) for g in args.target_generations.split(',')]
    print(f"Target generations: {target_generations}")
    
    # Parse display generations (for x-axis ticks)
    display_generations = [int(g) for g in args.display_generations.split(',')]
    print(f"Display generations (x-axis ticks): {display_generations}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Path to CSV file
    csv_path = os.path.join(args.output_dir, "batch_metrics.csv")
    
    # If vis_only flag is set, load data from CSV and generate visualizations
    if args.vis_only:
        print("Visualization only mode: Loading data from CSV and generating plots without recalculation")
        all_results = load_results_from_csv(csv_path, target_generations)
        
        if all_results:
            # Load real image metrics
            real_metrics = load_real_image_metrics(args.base_dir)
            
            # Generate visualization
            plot_output_path = os.path.join(args.output_dir, "batch_metrics_chart.png")
            plot_batch_metrics(all_results, plot_output_path, display_generations, include_std=True, real_metrics=real_metrics)
            print(f"Batch metrics chart saved to {plot_output_path}")
            
            return
        else:
            print(f"Failed to load data from CSV {csv_path}. Please check the file exists or run without --vis_only flag.")
            return
    
    # Delete specified generation entries from the CSV file
    if args.delete_gen is not None:
        print(f"Deleting generation {args.delete_gen} entries from CSV file...")
        delete_generation_from_csv(csv_path, args.delete_gen)
    
    # Delete generation 0 entries if recalculating
    if args.recalculate_gen0:
        print("Deleting generation 0 entries from CSV file for recalculation...")
        delete_generation_from_csv(csv_path, 0)
    
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
    
    # Load real image metrics
    real_metrics = load_real_image_metrics(args.base_dir)
    
    # For each group and generation, check if we need to analyze metrics
    for group in ["recursive", "finetune", "baseline"]:
        print(f"\nProcessing group: {group}")
        
        # Initialize results for this group
        group_results = {}
        
        for gen in target_generations:
            gen_str = str(gen)
            
            # Skip generation 0 if we already recalculated it
            if gen == 0 and args.recalculate_gen0:
                continue
            
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
            
            output_dir = os.path.join(args.base_dir, "gen_large", output_dir_name)
            
            # Check if directory has enough images
            if check_directory_has_enough_images(output_dir, args.batch_size):
                print(f"Analyzing {group} generation {gen} from {output_dir}")
                
                # Initialize results for this generation
                group_results[gen_str] = {}
                
                # Analyze random batches
                for trial in range(args.num_trials):
                    print(f"Analyzing trial {trial+1}/{args.num_trials}")
                    
                    # Analyze color metrics
                    results = analyze_random_batch(output_dir, batch_size=args.batch_size, max_images=args.max_images, gen=gen)
                    
                    # Calculate FID and CLIP scores
                    scores = calculate_random_batch_fid_and_clip(
                        output_dir, ref_stack, clip_model, clip_preprocess,
                        batch_size=args.batch_size, max_images=args.max_images, device=device, gen=gen
                    )
                    
                    # Store results
                    if results and scores:
                        group_results[gen_str][str(trial)] = {
                            'results': results,
                            'scores': scores
                        }
                
                # Update all_results with this group's results
                all_results[group] = group_results
                
                # Update visualization and CSV after each generation is analyzed
                print(f"\nUpdating visualization and CSV after analyzing {group} generation {gen}...")
                
                # Plot batch metrics with current data
                plot_output_path = os.path.join(args.output_dir, "batch_metrics_chart.png")
                plot_batch_metrics(all_results, plot_output_path, display_generations, include_std=True, real_metrics=real_metrics)
                print(f"Batch metrics chart updated at {plot_output_path}")
                
                # Save batch metrics to CSV
                save_batch_metrics_to_csv(all_results, csv_path, target_generations)
                print(f"Batch metrics CSV updated at {csv_path}")
                
                # Print current values for debugging
                print(f"\nCurrent values for {group} generation {gen}:")
                if gen_str in group_results:
                    # Calculate average FID and saturation across all batches
                    fid_values = []
                    saturation_values = []
                    for batch_idx, batch_data in group_results[gen_str].items():
                        if 'scores' in batch_data and batch_data['scores']['fid'] is not None:
                            fid_values.append(batch_data['scores']['fid'])
                        if 'results' in batch_data and batch_data['results']:
                            for result in batch_data['results']:
                                if 'saturation' in result:
                                    saturation_values.append(result['saturation'])
                    
                    avg_fid = np.mean(fid_values) if fid_values else "N/A"
                    avg_saturation = np.mean(saturation_values) if saturation_values else "N/A"
                    
                    print(f"  Avg FID = {avg_fid}, Avg Saturation = {avg_saturation}")
            else:
                print(f"Not enough images in {output_dir} for analysis")
    
    print("Analysis complete!")

def load_real_image_metrics(base_dir):
    """Load metrics for real images from the training set and real embeddings"""
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
            'saturation': 40.07,
            'color_std': 62.61,
            'contrast': 0.5766,
            'brightness': 114.27,
            'colorfulness': 41.03,
            'fid': 220.4891,
            'is_mean': 6.5722,
            'clip_score': 0.6099
        }
    
    # Load real embeddings data for RMG, L2M, and CLIP Variance
    real_npz_path = os.path.join("data", "embeddings", "CLIP_openai_clip-vit-base-patch32_embeddings_real.npz")
    if os.path.exists(real_npz_path):
        try:
            print(f"Loading real embedding metrics from {real_npz_path}")
            real_data = np.load(real_npz_path)
            
            # Extract image and text features using the correct field names
            img_features = real_data['image_embeddings']
            text_features = real_data['text_embeddings']
            
            # Calculate CLIP variance (embedding variance)
            real_metrics['clip_variance'] = float(np.mean(np.var(img_features, axis=0)))
            
            # Calculate L2M (L2 distance between mean embeddings)
            real_metrics['l2m'] = float(np.linalg.norm(np.mean(text_features, axis=0) - np.mean(img_features, axis=0)))
            
            # Calculate RMG (Relative Multimodal Gap)
            def cosine_dissim_rowwise(A, B):
                numerator = np.einsum('ij,ij->i', A, B)
                normA = np.linalg.norm(A, axis=1)
                normB = np.linalg.norm(B, axis=1)
                cos_sim = numerator / (normA * normB)
                return 1.0 - cos_sim  # dissimilarity
            
            N = img_features.shape[0]
            row_dissim_xy = cosine_dissim_rowwise(text_features, img_features)
            numerator = np.mean(row_dissim_xy)
            
            def sum_pairwise_cos_dissim(M):
                dot_mat = M @ M.T
                norms = np.linalg.norm(M, axis=1)
                norm_mat = np.outer(norms, norms)
                cos_sim_mat = dot_mat / norm_mat
                cos_dissim_mat = 1.0 - cos_sim_mat
                return np.sum(cos_dissim_mat)
            
            sum_dxx = sum_pairwise_cos_dissim(text_features)
            sum_dyy = sum_pairwise_cos_dissim(img_features)
            denom_part1 = (1.0 / (2.0 * N * (N - 1))) * (sum_dxx + sum_dyy)
            denom_part2 = numerator
            denominator = denom_part1 + denom_part2
            rmg_value = numerator / denominator
            
            real_metrics['rmg'] = float(rmg_value)
            
            print(f"Loaded real metrics - RMG: {real_metrics['rmg']:.4f}, L2M: {real_metrics['l2m']:.4f}, CLIP Variance: {real_metrics['clip_variance']:.4f}")
        except Exception as e:
            print(f"Error loading real embeddings: {e}")
            # Use default values if loading fails
            real_metrics['rmg'] = 0.7482
            real_metrics['l2m'] = 12.0547
            real_metrics['clip_variance'] = 0.1113
    else:
        print(f"Real embeddings file not found at {real_npz_path}, using default values")
        real_metrics['rmg'] = 0.7482
        real_metrics['l2m'] = 12.0547
        real_metrics['clip_variance'] = 0.1113
    
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

def load_results_from_csv(csv_path, target_generations=None):
    """Load results from CSV file and organize into the all_results structure.
    
    Args:
        csv_path: Path to the CSV file
        target_generations: Optional list of generations to include
        
    Returns:
        Dictionary with structure {group: {gen: {trial: {metric: value}}}}
    """
    if not os.path.exists(csv_path):
        print(f"CSV file {csv_path} does not exist.")
        return None
    
    import csv
    
    # Initialize all_results dictionary
    all_results = {
        "recursive": {},
        "finetune": {},
        "baseline": {}
    }
    
    # Define mapping from display group names to internal names
    group_mapping = {
        "Recursive Finetune": "recursive",
        "Real Finetune": "finetune",
        "Gen 0 Finetune": "baseline"
    }
    
    # Read the CSV file
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # Get generation, group, and batch
            gen = row['Generation']
            display_group = row['Group']
            batch_idx = row['Batch']
            
            # Skip if this generation is not in target_generations
            if target_generations is not None and int(gen) not in target_generations:
                continue
            
            # Map display group to internal group name
            if display_group in group_mapping:
                group = group_mapping[display_group]
            else:
                # Skip unknown groups
                continue
            
            # Initialize nested dictionaries if they don't exist
            if gen not in all_results[group]:
                all_results[group][gen] = {}
            
            if batch_idx not in all_results[group][gen]:
                all_results[group][gen][batch_idx] = {
                    'results': [],
                    'scores': {}
                }
            
            # Extract scores
            scores = all_results[group][gen][batch_idx]['scores']
            
            # Add score values if present
            for metric in ['FID', 'IS', 'CLIP_Score', 'RMG', 'L2M', 'CLIP_Variance']:
                csv_key = metric
                result_key = metric.lower()
                if metric == 'IS':
                    result_key = 'is_mean'
                
                if row[csv_key] and row[csv_key] != "":
                    scores[result_key] = float(row[csv_key])
            
            # Create a synthetic result for color metrics
            result = {
                'img_path': f"synthetic_{group}_{gen}_{batch_idx}",
                'saturation': float(row['Saturation']) if row['Saturation'] else 0,
                'contrast': float(row['Contrast']) if row['Contrast'] else 0,
                'brightness': float(row['Brightness']) if row['Brightness'] else 0,
                'color_std': float(row['Color_Std']) if row['Color_Std'] else 0
            }
            
            # Add result to results list
            all_results[group][gen][batch_idx]['results'].append(result)
    
    # Check if we loaded any data
    has_data = False
    for group in all_results:
        if all_results[group]:
            has_data = True
            break
    
    if not has_data:
        print(f"No relevant data found in CSV file {csv_path}.")
        return None
    
    return all_results

if __name__ == "__main__":
    main() 