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
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, InterpolationMode

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

def collect_data_for_generations(base_dir, target_generations=None, include_groups=True):
    """Collect color distribution data for specified generations and groups.
    
    Args:
        base_dir: Base directory containing the image folders
        target_generations: List of generations to analyze
        include_groups: Whether to include finetune and baseline groups
    """
    # Dictionary to store results by generation and group
    results = {}
    
    # Use the gen folder instead of steps_gen
    gen_dir = os.path.join(base_dir, "gen")
    
    # Find all generation folders
    all_folders = glob.glob(os.path.join(gen_dir, "sd_to_sd_cfg_*_steps_50_gen_*"))
    
    # Extract all available generations
    all_generations = set()
    for folder in all_folders:
        gen_num = extract_generation_number(folder)
        if gen_num is not None:
            all_generations.add(gen_num)
    
    # If no target generations specified, use all available generations
    if target_generations is None:
        target_generations = sorted(all_generations)
    
    # Initialize results dictionary for all target generations
    for gen in target_generations:
        results[str(gen)] = []  # Regular recursive finetune - use string keys for consistency
        if include_groups:
            results[f"{gen}_finetune"] = []  # Real finetune
            results[f"{gen}_baseline"] = []  # Gen 0 finetune (baseline)
    
    # Group folders by generation and type
    for folder in all_folders:
        folder_name = os.path.basename(folder)
        
        # Skip if not a valid folder
        if not folder_name.startswith("sd_to_sd_cfg_"):
            continue
        
        gen_num = extract_generation_number(folder)
        if gen_num is not None and gen_num in target_generations:
            # Determine the group based on folder name
            if "_finetune" in folder_name:
                group_key = f"{gen_num}_finetune"
            elif "_baseline" in folder_name:
                group_key = f"{gen_num}_baseline"
            else:
                group_key = str(gen_num)  # Convert to string for consistency
            
            # Skip if we're not including groups and this is a group folder
            if not include_groups and group_key != str(gen_num):
                continue
            
            # Process images in this folder
            image_files = glob.glob(os.path.join(folder, "*.jpg"))
            print(f"  Found {len(image_files)} images in {folder_name}")
            
            # Limit to 100 images per generation/group
            image_files = image_files[:100]
            
            for img_path in image_files:
                color_data = analyze_color_distribution(img_path)
                if color_data:
                    if group_key in results:
                        results[group_key].append(color_data)
    
    # Print summary
    for key in sorted(results.keys()):
        if key.isdigit():
            group_name = "Recursive Finetune"
            gen_num = key
        elif "_finetune" in key:
            group_name = "Real Finetune"
            gen_num = key.split('_')[0]
        elif "_baseline" in key:
            group_name = "Gen 0 Finetune"
            gen_num = key.split('_')[0]
        else:
            group_name = "Unknown"
            gen_num = key
        
        print(f"Collected data for {group_name} {gen_num}: {len(results[key])} images")
    
    return results

def calculate_fid_and_clip_scores(base_dir, target_generations, include_groups=True):
    """Calculate FID and CLIP scores for each generation and group."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Results dictionary
    scores = {}
    for gen in target_generations:
        scores[str(gen)] = {'fid': None, 'is_mean': None, 'is_std': None, 'clip_score': None}
        if include_groups:
            scores[f"{gen}_finetune"] = {'fid': None, 'is_mean': None, 'is_std': None, 'clip_score': None}
            scores[f"{gen}_baseline"] = {'fid': None, 'is_mean': None, 'is_std': None, 'clip_score': None}
    
    try:
        # Load CLIP model
        print("Loading CLIP model...")
        model_name = "ViT-B/32"
        clip_model, clip_preprocess = clip.load(model_name, device=device, download_root=os.path.expanduser("~/.cache/clip"))
        print(f"CLIP model {model_name} loaded successfully")
    except Exception as e:
        print(f"Error loading CLIP model: {str(e)}")
        clip_model = None
        clip_preprocess = None
    
    # Reference folder for original COCO images
    ref_folder = os.path.join(base_dir, "train2014")
    
    # Define transforms for FID calculation
    transform = Compose([
        Resize(299, interpolation=InterpolationMode.BILINEAR),
        CenterCrop(299),
        ToTensor(),
    ])
    
    # Load reference images (first 1000 for efficiency)
    print("Loading reference images...")
    ref_images = []
    ref_files = sorted(glob.glob(os.path.join(ref_folder, "*.jpg")))[:1000]
    
    for img_path in ref_files:
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
        return scores
    
    ref_stack = torch.stack(ref_images).to(device)
    
    # Use the gen folder
    gen_dir = os.path.join(base_dir, "gen")
    
    # Process each generation and group
    for gen in target_generations:
        # Define the groups to process
        groups_to_process = [(str(gen), "")]  # Regular recursive finetune
        if include_groups:
            groups_to_process.extend([
                (f"{gen}_finetune", "_finetune"),  # Real finetune
                (f"{gen}_baseline", "_baseline")   # Gen 0 finetune (baseline)
            ])
        
        for group_key, group_suffix in groups_to_process:
            print(f"Calculating FID and CLIP scores for {group_key}...")
            
            # Find folders for this generation and group
            pattern = f"sd_to_sd_cfg_*_steps_50_gen_{gen}{group_suffix}"
            gen_folders = glob.glob(os.path.join(gen_dir, pattern))
            
            if not gen_folders:
                print(f"No folders found for {group_key}")
                continue
            
            # Load generated images (limit to 100 total)
            gen_images = []
            gen_pil_images = []  # Store original PIL images for CLIP
            max_images = 100
            
            for folder in gen_folders:
                img_files = sorted(glob.glob(os.path.join(folder, "*.jpg")))
                remaining = max_images - len(gen_images)
                
                if remaining <= 0:
                    break
                    
                img_files = img_files[:remaining]
                print(f"  Adding {len(img_files)} images from {os.path.basename(folder)}")
                
                for img_path in img_files:
                    try:
                        img = Image.open(img_path).convert('RGB')
                        gen_pil_images.append(img)  # Store the PIL image for CLIP
                        
                        img_tensor = transform(img)
                        # Scale to [0, 255] and convert to uint8 for FID
                        img_tensor = (img_tensor * 255).to(torch.uint8)
                        gen_images.append(img_tensor)
                    except Exception as e:
                        print(f"Error loading generated image {img_path}: {e}")
            
            if not gen_images:
                print(f"No generated images found for {group_key}")
                continue
            
            print(f"  Loaded {len(gen_images)} images for {group_key}")
            
            gen_stack = torch.stack(gen_images).to(device)
            
            # Calculate FID
            try:
                fid = FrechetInceptionDistance(normalize=True).to(device)
                
                # Update FID with batches to avoid OOM
                batch_size = 32
                for i in range(0, len(ref_stack), batch_size):
                    batch = ref_stack[i:i+batch_size]
                    fid.update(batch, real=True)
                
                for i in range(0, len(gen_stack), batch_size):
                    batch = gen_stack[i:i+batch_size]
                    fid.update(batch, real=False)
                
                scores[group_key]['fid'] = float(fid.compute())
                print(f"{group_key} FID: {scores[group_key]['fid']:.2f}")
                
                del fid
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error calculating FID for {group_key}: {e}")
            
            # Calculate Inception Score
            try:
                inception_score = InceptionScore(normalize=True).to(device)
                
                # Process in batches
                for i in range(0, len(gen_stack), batch_size):
                    batch = gen_stack[i:i+batch_size]
                    inception_score.update(batch)
                
                is_mean, is_std = inception_score.compute()
                scores[group_key]['is_mean'] = float(is_mean)
                scores[group_key]['is_std'] = float(is_std)
                print(f"{group_key} IS: {scores[group_key]['is_mean']:.2f} ± {scores[group_key]['is_std']:.2f}")
                
                del inception_score
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error calculating IS for {group_key}: {e}")
            
            # Calculate CLIP score if model is available
            if clip_model is not None:
                try:
                    # Simple CLIP score calculation (average cosine similarity between image and text embeddings)
                    clip_scores = []
                    
                    # Process in smaller batches
                    batch_size = 16
                    for i in range(0, len(gen_pil_images), batch_size):
                        end_idx = min(i + batch_size, len(gen_pil_images))
                        batch = gen_pil_images[i:end_idx]
                        
                        # Create a simple caption for each image
                        captions = [f"An image from {group_key}"] * len(batch)
                        
                        # Preprocess images and text
                        processed_images = torch.stack([clip_preprocess(img) for img in batch]).to(device)
                        text_tokens = clip.tokenize(captions).to(device)
                        
                        # Get embeddings
                        with torch.no_grad():
                            image_features = clip_model.encode_image(processed_images)
                            text_features = clip_model.encode_text(text_tokens)
                        
                        # Normalize features
                        image_features = image_features / image_features.norm(dim=1, keepdim=True)
                        text_features = text_features / text_features.norm(dim=1, keepdim=True)
                        
                        # Calculate similarity
                        similarities = (image_features @ text_features.T).diag()  # Now in range -1 to 1
                        # Normalize to 0-1 range
                        similarities = (similarities + 1) / 2
                        clip_scores.extend(similarities.cpu().numpy())
                    
                    scores[group_key]['clip_score'] = float(np.mean(clip_scores))
                    print(f"{group_key} CLIP Score: {scores[group_key]['clip_score']:.2f}")
                    
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Error calculating CLIP score for {group_key}: {e}")
    
    return scores

def calculate_real_image_metrics(base_dir):
    """Calculate metrics for real images from the training set."""
    print("Calculating metrics for real images from the training set...")
    
    # Path to real images
    ref_folder = os.path.join(base_dir, "train2014")
    
    # Get first 100 images
    real_image_files = sorted(glob.glob(os.path.join(ref_folder, "*.jpg")))[:100]
    if not real_image_files:
        print("No real images found!")
        return None, None
    
    print(f"Found {len(real_image_files)} real images")
    
    # Analyze color distribution
    real_results = []
    for img_path in real_image_files:
        color_data = analyze_color_distribution(img_path)
        if color_data:
            real_results.append(color_data)
    
    # Calculate FID and CLIP scores
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # For real images, FID would be 0 by definition
    real_scores = {'fid': 0.0, 'is_mean': None, 'is_std': None, 'clip_score': None}
    
    # Calculate Inception Score for real images
    try:
        # Load images
        transform = Compose([
            Resize(299, interpolation=InterpolationMode.BILINEAR),
            CenterCrop(299),
            ToTensor(),
        ])
        
        real_tensors = []
        for img_path in real_image_files:
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)
                # Scale to [0, 255] and convert to uint8 for IS
                img_tensor = (img_tensor * 255).to(torch.uint8)
                real_tensors.append(img_tensor)
            except Exception as e:
                print(f"Error loading real image {img_path}: {e}")
        
        if real_tensors:
            real_stack = torch.stack(real_tensors).to(device)
            
            # Calculate Inception Score
            inception_score = InceptionScore(normalize=True).to(device)
            
            # Process in batches
            batch_size = 32
            for i in range(0, len(real_stack), batch_size):
                batch = real_stack[i:i+batch_size]
                inception_score.update(batch)
            
            is_mean, is_std = inception_score.compute()
            real_scores['is_mean'] = float(is_mean)
            real_scores['is_std'] = float(is_std)
            print(f"Real images IS: {real_scores['is_mean']:.2f} ± {real_scores['is_std']:.2f}")
            
            del inception_score
            torch.cuda.empty_cache()
            
            # Calculate CLIP score for real images
            try:
                model_name = "ViT-B/32"
                clip_model, clip_preprocess = clip.load(model_name, device=device, download_root=os.path.expanduser("~/.cache/clip"))
                
                clip_scores = []
                batch_size = 16
                
                for i in range(0, len(real_tensors), batch_size):
                    end_idx = min(i + batch_size, len(real_tensors))
                    batch = [img.to(torch.float32) / 255.0 for img in real_tensors[i:end_idx]]
                    
                    # Create captions
                    captions = ["A photograph"] * len(batch)
                    
                    # Preprocess images and text
                    processed_images = torch.stack([clip_preprocess(img) for img in batch]).to(device)
                    text_tokens = clip.tokenize(captions).to(device)
                    
                    # Get embeddings
                    with torch.no_grad():
                        image_features = clip_model.encode_image(processed_images)
                        text_features = clip_model.encode_text(text_tokens)
                    
                    # Normalize features
                    image_features = image_features / image_features.norm(dim=1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=1, keepdim=True)
                    
                    # Calculate similarity (normalize to 0-1 range)
                    similarities = (image_features @ text_features.T).diag()
                    similarities = (similarities + 1) / 2  # Convert from -1,1 to 0,1
                    clip_scores.extend(similarities.cpu().numpy())
                
                real_scores['clip_score'] = float(np.mean(clip_scores))
                print(f"Real images CLIP Score: {real_scores['clip_score']:.2f}")
                
                del clip_model
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error calculating CLIP score for real images: {e}")
    except Exception as e:
        print(f"Error calculating IS for real images: {e}")
    
    return real_results, real_scores

def plot_metrics_by_group(results, scores, output_path, target_generations=[0, 5, 10], real_results=None, real_scores=None):
    """Create line charts for each metric, showing trends across generations.
    
    Args:
        results: Dictionary of image analysis results by generation and group
        scores: Dictionary of FID, IS, and CLIP scores by generation and group
        output_path: Path to save the output image
        target_generations: List of specific generations to plot
        real_results: Results for real images (for reference line)
        real_scores: Scores for real images (for reference line)
    """
    # Convert target_generations to strings for consistent key lookup
    target_generations_str = [str(gen) for gen in target_generations]
    
    # Define the groups and their display names
    groups = [
        {"key": "", "name": "Recursive Finetune", "color": PRIMARY},
        {"key": "_finetune", "name": "Real Finetune", "color": SECONDARY},
        {"key": "_baseline", "name": "Gen 0 Finetune", "color": TERTIARY}
    ]
    
    # Define all the metrics to plot
    all_metrics = [
        # Color metrics 
        {'name': 'Saturation', 'key': 'saturation', 'source': 'results', 'group': 'Color'},
        {'name': 'Contrast', 'key': 'contrast', 'source': 'results', 'group': 'Color'},
        {'name': 'Brightness', 'key': 'brightness', 'source': 'results', 'group': 'Color'},
        {'name': 'Colorfulness', 'key': 'colorfulness', 'source': 'results', 'group': 'Color'},
        {'name': 'Color Std', 'key': 'color_std', 'source': 'results', 'group': 'Color'},
        
        # Generation quality metrics
        {'name': 'FID', 'key': 'fid', 'source': 'scores', 'group': 'Generation'},
        {'name': 'IS', 'key': 'is_mean', 'source': 'scores', 'group': 'Generation'},
        {'name': 'CLIP Score', 'key': 'clip_score', 'source': 'scores', 'group': 'Generation'}
    ]
    
    # Group metrics by category for file organization
    metric_groups = {}
    for metric in all_metrics:
        group = metric['group']
        if group not in metric_groups:
            metric_groups[group] = []
        metric_groups[group].append(metric)
    
    # Calculate reference values from real images
    real_ref_values = {}
    if real_results:
        for metric in all_metrics:
            if metric['source'] == 'results':
                key = metric['key']
                # For color_std, use color_variance in real_results if color_std is not available
                if key == 'color_std':
                    values = [data.get('color_std', data.get('color_variance')) for data in real_results 
                             if (key in data and data[key] is not None) or 
                             ('color_variance' in data and data['color_variance'] is not None)]
                else:
                    values = [data[key] for data in real_results if key in data and data[key] is not None]
                
                if values:
                    real_ref_values[key] = np.mean(values)
    
    if real_scores:
        for metric in all_metrics:
            if metric['source'] == 'scores':
                key = metric['key']
                if key in real_scores and real_scores[key] is not None:
                    real_ref_values[key] = real_scores[key]
    
    # Create a subplot for each metric
    for group_name, group_metrics in metric_groups.items():
        # Calculate number of rows and columns for subplots
        n_metrics = len(group_metrics)
        n_cols = min(3, n_metrics)  # At most 3 columns
        n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division
        
        # Create figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), squeeze=False)
        fig.tight_layout(pad=3.0)
        
        # Flatten axes for easier indexing
        axes = axes.flatten()
        
        # Use consistent line style for all metrics
        line_style = '-'
        marker_style = 'o'
        line_width = 2
        marker_size = 8
        
        # For each metric, create a line plot
        for i, metric in enumerate(group_metrics):
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
                group_suffix = group["key"]
                group_color = group["color"]
                group_name = group["name"]
                
                # For each generation, collect and plot the metric value
                x_values = []
                y_values = []
                
                for gen_idx, gen_str in enumerate(target_generations_str):
                    gen = target_generations[gen_idx]  # Original integer for x-axis
                    
                    # Construct the key for this generation and group
                    if group_suffix:
                        key = f"{gen_str}{group_suffix}"
                    else:
                        key = gen_str
                    
                    if source == 'scores':
                        # Get from scores dictionary
                        if key in scores and scores[key][metric_key] is not None:
                            x_values.append(gen)
                            # Normalize CLIP score to 0-1 range if not already
                            if metric_key == 'clip_score' and scores[key][metric_key] > 1:
                                y_values.append(scores[key][metric_key] / 100.0)
                            else:
                                y_values.append(scores[key][metric_key])
                    else:
                        # Calculate from results dictionary
                        if key in results and results[key]:
                            # For color_std, use color_variance if color_std is not available
                            if metric_key == 'color_std':
                                metric_data = [data.get('color_std', data.get('color_variance')) for data in results[key] 
                                             if (metric_key in data and data[metric_key] is not None) or 
                                             ('color_variance' in data and data['color_variance'] is not None)]
                            else:
                                metric_data = [data[metric_key] for data in results[key] 
                                             if metric_key in data and data[metric_key] is not None]
                            
                            if metric_data:
                                x_values.append(gen)
                                y_values.append(np.mean(metric_data))
                
                if x_values and y_values:
                    # Plot this group
                    line, = ax.plot(x_values, y_values, marker=marker_style, linestyle=line_style, 
                                   linewidth=line_width, markersize=marker_size, color=group_color)
                    lines.append(line)
                    labels.append(group_name)
            
            # Add horizontal reference line for real images if available
            if metric_key in real_ref_values:
                ref_value = real_ref_values[metric_key]
                if metric_key == 'clip_score' and ref_value > 1:
                    ref_value = ref_value / 100.0  # Normalize to 0-1 range
                
                ax.axhline(y=ref_value, color=GREY_500, linestyle='--', linewidth=line_width-0.5)
                ax.text(max(target_generations), ref_value, 'Real', va='bottom', ha='right', 
                        color=GREY_700, fontsize=10, fontweight='bold')
            
            # Set x-axis ticks for all generations
            ax.set_xticks(list(range(0, 11, 2)))  # Show 0, 2, 4, 6, 8, 10
            ax.set_xticklabels([str(x) for x in range(0, 11, 2)])
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # Add legend only to the first subplot
            if i == 0 and lines:
                ax.legend(lines, labels, loc='best', fontsize=10)
            
            # Set axis labels (no x-axis title as requested)
            ax.set_ylabel(metric['name'], fontsize=12)
        
        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        
        # Save the figure for this group
        group_output_path = os.path.join(os.path.dirname(output_path), f"{group_name.lower()}_metrics_line.png")
        plt.savefig(group_output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"{group_name} metrics chart saved to {group_output_path}")
    
    # Create a single combined figure with all metrics
    # Calculate number of rows and columns for subplots
    n_metrics = len(all_metrics)
    n_cols = min(4, n_metrics)  # At most 4 columns for the combined view
    n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.5 * n_rows), squeeze=False)
    
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
            group_suffix = group["key"]
            group_color = group["color"]
            group_name = group["name"]
            
            # For each generation, collect and plot the metric value
            x_values = []
            y_values = []
            
            for gen_idx, gen_str in enumerate(target_generations_str):
                gen = target_generations[gen_idx]  # Original integer for x-axis
                
                # Construct the key for this generation and group
                if group_suffix:
                    key = f"{gen_str}{group_suffix}"
                else:
                    key = gen_str
                
                if source == 'scores':
                    # Get from scores dictionary
                    if key in scores and scores[key][metric_key] is not None:
                        x_values.append(gen)
                        # Normalize CLIP score to 0-1 range if not already
                        if metric_key == 'clip_score' and scores[key][metric_key] > 1:
                            y_values.append(scores[key][metric_key] / 100.0)
                        else:
                            y_values.append(scores[key][metric_key])
                else:
                    # Calculate from results dictionary
                    if key in results and results[key]:
                        # For color_std, use color_variance if color_std is not available
                        if metric_key == 'color_std':
                            metric_data = [data.get('color_std', data.get('color_variance')) for data in results[key] 
                                         if (metric_key in data and data[metric_key] is not None) or 
                                         ('color_variance' in data and data['color_variance'] is not None)]
                        else:
                            metric_data = [data[metric_key] for data in results[key] 
                                         if metric_key in data and data[metric_key] is not None]
                        
                        if metric_data:
                            x_values.append(gen)
                            y_values.append(np.mean(metric_data))
            
            if x_values and y_values:
                # Plot this group
                line, = ax.plot(x_values, y_values, marker=marker_style, linestyle=line_style, 
                               linewidth=line_width, markersize=marker_size, color=group_color)
                lines.append(line)
                labels.append(group_name)
        
        # Add horizontal reference line for real images if available
        if metric_key in real_ref_values:
            ref_value = real_ref_values[metric_key]
            if metric_key == 'clip_score' and ref_value > 1:
                ref_value = ref_value / 100.0
                
            ax.axhline(y=ref_value, color=GREY_500, linestyle='--', linewidth=line_width-0.5)
            ax.text(max(target_generations), ref_value, 'Real', va='bottom', ha='right', 
                    color=GREY_700, fontsize=10, fontweight='bold')
        
        # Set x-axis ticks for all generations
        ax.set_xticks(list(range(0, 11, 2)))  # Show 0, 2, 4, 6, 8, 10
        ax.set_xticklabels([str(x) for x in range(0, 11, 2)])
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Add legend only to the first subplot
        if i == 0 and lines:
            ax.legend(lines, labels, loc='best', fontsize=10)
        
        # Set axis labels (no x-axis title as requested)
        ax.set_ylabel(metric['name'], fontsize=10)
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    # Adjust layout
    plt.tight_layout(pad=2.0)
    
    # Save the combined figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined metrics chart saved to {output_path}")

def save_metrics_to_csv(results, scores, output_path):
    """Save all metrics to a CSV file."""
    # Get all keys (generations and groups)
    all_keys = sorted(results.keys(), key=lambda x: (int(x.split('_')[0]), x))
    
    with open(output_path, 'w') as f:
        # Write header
        f.write("Generation,Group,Images,Avg_R,Avg_G,Avg_B,Saturation,Color_Std,Contrast,Brightness,Colorfulness,FID,IS,CLIP_Score\n")
        
        # Write data for each generation and group
        for key in all_keys:
            gen_data = results[key]
            if not gen_data:
                continue
            
            # Determine generation number and group
            if key.isdigit():
                gen = key
                group = "Recursive Finetune"
            else:
                # Extract generation number and group from string key
                parts = key.split('_')
                gen = parts[0]
                if "finetune" in key:
                    group = "Real Finetune"
                elif "baseline" in key:
                    group = "Gen 0 Finetune"
                else:
                    group = "Unknown"
            
            # Calculate averages for color metrics
            avg_rgb = np.mean([data['avg_rgb'] for data in gen_data], axis=0)
            
            saturation_values = [data['saturation'] for data in gen_data if 'saturation' in data]
            avg_saturation = np.mean(saturation_values) if saturation_values else 0
            
            std_values = [data['color_std'] for data in gen_data if 'color_std' in data]
            avg_std = np.mean(std_values) if std_values else 0
            
            contrast_values = [data['contrast'] for data in gen_data if 'contrast' in data and data['contrast'] is not None]
            avg_contrast = np.mean(contrast_values) if contrast_values else 0
            
            brightness_values = [data['brightness'] for data in gen_data if 'brightness' in data and data['brightness'] is not None]
            avg_brightness = np.mean(brightness_values) if brightness_values else 0
            
            colorfulness_values = [data['colorfulness'] for data in gen_data if 'colorfulness' in data and data['colorfulness'] is not None]
            avg_colorfulness = np.mean(colorfulness_values) if colorfulness_values else 0
            
            # Get FID, IS, and CLIP scores
            fid = scores[key]['fid'] if key in scores and scores[key]['fid'] is not None else ""
            is_mean = scores[key]['is_mean'] if key in scores and scores[key]['is_mean'] is not None else ""
            clip_score = scores[key]['clip_score'] if key in scores and scores[key]['clip_score'] is not None else ""
            
            # Write row
            f.write(f"{gen},{group},{len(gen_data)},{avg_rgb[0]:.1f},{avg_rgb[1]:.1f},{avg_rgb[2]:.1f},{avg_saturation:.1f},{avg_std:.1f},{avg_contrast:.3f},{avg_brightness:.1f},{avg_colorfulness:.1f},{fid},{is_mean},{clip_score}\n")
    
    print(f"Metrics saved to {output_path}")

def main():
    # Base directory containing the image folders
    base_dir = os.path.join("data", "coco")
    
    # Include all generations from 0 to 10
    target_generations = list(range(11))  # 0 to 10 inclusive
    
    print(f"Analyzing generations: {target_generations}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join("vis", "t2i", "metrics")
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect data for target generations and groups (100 images per generation/group)
    print(f"Collecting data for generations {target_generations} and groups...")
    results = collect_data_for_generations(base_dir, target_generations, include_groups=True)
    
    # Calculate metrics for real images from training set (first 100 images)
    real_results, real_scores = calculate_real_image_metrics(base_dir)
    
    # Calculate FID and CLIP scores (100 images per generation/group)
    print(f"Calculating FID and CLIP scores for generations {target_generations} and groups...")
    scores = calculate_fid_and_clip_scores(base_dir, target_generations, include_groups=True)
    
    # Plot metrics grouped by type with reference lines
    grouped_output_path = os.path.join(output_dir, "metrics_by_group.png")
    plot_metrics_by_group(results, scores, grouped_output_path, target_generations, real_results, real_scores)
    
    # Save numerical results to CSV
    csv_output_path = os.path.join(output_dir, "metrics_results.csv")
    save_metrics_to_csv(results, scores, csv_output_path)
    
    print(f"Analysis complete. All charts saved to: {output_dir}")
    print(f"Metrics by group chart: {grouped_output_path}")

if __name__ == "__main__":
    main() 