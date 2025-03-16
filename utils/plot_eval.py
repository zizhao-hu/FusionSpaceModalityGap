import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from tqdm import tqdm
import re
import json
import sys
import clip
import random
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "clipscore"))
from clipscore import get_clip_score, extract_all_images, get_refonlyclipscore
import sklearn
from packaging import version

# Import color constants
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from utils.colors import *

def extract_image_id(filename):
    """Extract COCO image ID from filename"""
    match = re.search(r'COCO_train2014_(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def get_image_ids_from_folder(folder_path):
    """Get sorted list of image IDs from a folder"""
    image_ids = []
    for f in os.listdir(folder_path):
        if f.endswith(('.jpg', '.png')):
            image_id = extract_image_id(f)
            if image_id is not None:
                image_ids.append(image_id)
    return sorted(image_ids)

def load_and_preprocess_images(folder_path, target_image_ids=None):
    """Load and preprocess specific images from a folder"""
    images = []
    if not os.path.exists(folder_path):
        print(f"Warning: Folder not found: {folder_path}")
        return None

    # Create mapping of image IDs to filenames
    id_to_file = {}
    for f in sorted(os.listdir(folder_path)):  # Sort to ensure consistent ordering
        if f.endswith(('.jpg', '.png')):
            image_id = extract_image_id(f)
            if image_id is not None:
                id_to_file[image_id] = f

    # If no target IDs provided, use first 1000 available IDs
    if target_image_ids is None:
        target_image_ids = sorted(list(id_to_file.keys()))[:1000]
        print(f"Using first {len(target_image_ids)} image IDs from {folder_path}")

    # Define transforms for FID calculation
    transform = Compose([
        Resize(299, interpolation=InterpolationMode.BILINEAR),
        CenterCrop(299),
        ToTensor(),
    ])

    # Load only the target images in order
    valid_count = 0
    for image_id in tqdm(target_image_ids, desc=f"Loading images from {os.path.basename(folder_path)}"):
        if image_id not in id_to_file:
            print(f"Warning: Image ID {image_id} not found in {folder_path}")
            continue

        try:
            img_path = os.path.join(folder_path, id_to_file[image_id])
            if os.path.getsize(img_path) > 1024:  # Skip empty or corrupt images
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)
                # Scale to [0, 255] and convert to uint8 for FID
                img_tensor = (img_tensor * 255).to(torch.uint8)
                images.append(img_tensor)
                valid_count += 1
        except Exception as e:
            print(f"Error loading {id_to_file[image_id]}: {str(e)}")
            continue

    print(f"Successfully loaded {valid_count} valid images from {folder_path}")
    if not images:
        print(f"No valid images found in {folder_path}")
        return None

    return images

def load_coco_captions(annotation_file="data/coco/annotations/captions_train2014.json"):
    """Load COCO captions from json file"""
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # Create mapping of image_id to first caption only
    image_id_to_caption = {}
    for ann in annotations['annotations']:
        image_id = ann['image_id']
        if image_id not in image_id_to_caption:  # Only take the first caption
            image_id_to_caption[image_id] = ann['caption']
    
    return image_id_to_caption

def prepare_clipscore_inputs(folder_path, image_ids, captions):
    """Prepare inputs in the format expected by CLIPScore"""
    candidates = {}
    references = {}
    
    # Create mapping of image IDs to filenames
    id_to_file = {}
    for f in os.listdir(folder_path):
        if f.endswith(('.jpg', '.png')):
            image_id = extract_image_id(f)
            if image_id is not None:
                id_to_file[image_id] = f
    
    # Create candidates and references dictionaries
    for image_id, caption in zip(image_ids, captions):
        if image_id in id_to_file:
            img_path = os.path.join(folder_path, id_to_file[image_id])
            candidates[img_path] = caption
            references[img_path] = caption  # Using same caption as reference
    
    return candidates, references

def evaluate_folder(ref_images, folder_path, target_image_ids, clip_model=None, clip_preprocess=None):
    """Calculate FID, IS, and CLIP scores for a folder of images"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load generated images
    gen_images = load_and_preprocess_images(folder_path, target_image_ids)
    if not gen_images:
        return None
    
    results = {}
    
    # Calculate FID
    try:
        fid = FrechetInceptionDistance(normalize=True).to(device)
        # Stack images and ensure they're on CPU first
        ref_stack = torch.stack(ref_images).cpu()
        gen_stack = torch.stack(gen_images).cpu()
        
        # Update FID with batches to avoid OOM
        batch_size = 32
        for i in range(0, len(ref_stack), batch_size):
            batch = ref_stack[i:i+batch_size].to(device)
            fid.update(batch, real=True)
        
        for i in range(0, len(gen_stack), batch_size):
            batch = gen_stack[i:i+batch_size].to(device)
            fid.update(batch, real=False)
        
        results['fid'] = float(fid.compute())
        del fid
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error calculating FID: {str(e)}")
        results['fid'] = float('nan')
    
    # Calculate Inception Score
    try:
        inception_score = InceptionScore(normalize=True).to(device)
        # Process in batches
        for i in range(0, len(gen_images), batch_size):
            batch = torch.stack(gen_images[i:i+batch_size]).to(device)
            inception_score.update(batch)
        
        is_mean, is_std = inception_score.compute()
        results['is_mean'] = float(is_mean)
        results['is_std'] = float(is_std)
        del inception_score
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error calculating IS: {str(e)}")
        results['is_mean'] = float('nan')
        results['is_std'] = float('nan')
    
    # Calculate CLIP scores if model is available
    if clip_model is not None and clip_preprocess is not None:
        try:
            # Load captions from COCO annotations
            image_id_to_caption = load_coco_captions()
            
            # Create candidates dictionary in the format expected by CLIPScore
            candidates = {}
            references = {}
            
            # Create mapping of image IDs to filenames
            id_to_file = {}
            for f in os.listdir(folder_path):
                if f.endswith(('.jpg', '.png')):
                    image_id = extract_image_id(f)
                    if image_id is not None:
                        id_to_file[image_id] = f
            
            # Prepare candidates and references dictionaries
            ref_folder = "data/coco/train2014"
            for image_id in target_image_ids:
                if image_id in id_to_file and image_id in image_id_to_caption:
                    # Generated image path for candidate
                    gen_img_path = os.path.abspath(os.path.join(folder_path, id_to_file[image_id]))
                    # Reference image path
                    ref_img_path = os.path.abspath(os.path.join(ref_folder, f"COCO_train2014_{image_id:012d}.jpg"))
                    
                    if os.path.exists(gen_img_path) and os.path.exists(ref_img_path):
                        candidates[gen_img_path] = image_id_to_caption[image_id]
                        references[gen_img_path] = ref_img_path
            
            print(f"Number of candidates: {len(candidates)}")
            print(f"Sample candidate entry: {next(iter(candidates.items()))}")
            print(f"Sample reference entry: {next(iter(references.items()))}")
            
            try:
                print("Calculating CLIPScore...")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                # Extract features for generated images (candidates)
                candidate_features = extract_all_images(list(candidates.keys()), clip_model, device)
                
                # Extract features for reference images
                reference_features = extract_all_images(list(references.values()), clip_model, device)
                
                # Get image-text CLIPScore (no references)
                clip_score, per_instance_scores, _ = get_clip_score(
                    clip_model, 
                    candidate_features,  # Pass pre-computed image features
                    list(candidates.values()),  # Pass just the captions
                    device
                )
                results['clip_score'] = float(clip_score)
                
                # Calculate RefCLIPScore using reference images
                if version.parse(np.__version__) < version.parse('1.21'):
                    candidate_features = sklearn.preprocessing.normalize(candidate_features, axis=1)
                    reference_features = sklearn.preprocessing.normalize(reference_features, axis=1)
                else:
                    candidate_features = candidate_features / np.sqrt(np.sum(candidate_features**2, axis=1, keepdims=True))
                    reference_features = reference_features / np.sqrt(np.sum(reference_features**2, axis=1, keepdims=True))
                
                # Calculate similarity between candidate and reference images
                similarities = np.sum(candidate_features * reference_features, axis=1)
                refclip_score = np.mean(similarities)
                results['refclip_score'] = float(refclip_score)
                
                print(f"CLIPScore: {results['clip_score']}")
                print(f"RefCLIPScore: {results['refclip_score']}")
                
            except Exception as e:
                print(f"Error calculating CLIPScore: {str(e)}")
                print("Full error:", e.__class__.__name__, str(e))
                import traceback
                traceback.print_exc()
                results['clip_score'] = float('nan')
                results['refclip_score'] = float('nan')
                
        except Exception as e:
            print(f"Error in CLIPScore preparation: {str(e)}")
            results['clip_score'] = float('nan')
            results['refclip_score'] = float('nan')
    else:
        results['clip_score'] = float('nan')
        results['refclip_score'] = float('nan')
    
    return results

def plot_metrics(all_metrics, output_path="evaluation_results.png"):
    """Plot evaluation metrics for different CFG values"""
    # Extract metrics
    cfg_values = sorted(all_metrics.keys())
    fid_values = [all_metrics[cfg]['fid'] for cfg in cfg_values]
    is_values = [all_metrics[cfg]['is'] for cfg in cfg_values]
    clip_values = [all_metrics[cfg]['clip'] for cfg in cfg_values]
    refclip_values = [all_metrics[cfg]['refclip'] for cfg in cfg_values]
    
    # Create figure with GridSpec for better control over subplot alignment
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.25, hspace=0.3)
    
    fig.suptitle('Image Generation Quality Metrics vs. CFG Scale', 
                fontsize=20, fontweight='bold', color=GREY_900, y=0.98)
    
    # Create a list to store all axes for later synchronization
    all_axes = []
    
    # FID plot (lower is better)
    ax1 = fig.add_subplot(gs[0, 0])
    all_axes.append(ax1)
    ax1.set_facecolor('white')
    ax1.plot(cfg_values, fid_values, 'o-', color=PRIMARY, linewidth=2.5, markersize=10)
    ax1.set_title('FID Score (lower is better)', fontsize=16, fontweight='bold', color=GREY_900, pad=10)
    ax1.set_xlabel('CFG Scale', fontsize=14, fontweight='bold', color=GREY_800)
    ax1.set_ylabel('FID', fontsize=14, fontweight='bold', color=GREY_800)
    ax1.grid(True, linestyle='--', alpha=0.3, color=GREY_300)
    ax1.tick_params(axis='both', which='major', labelsize=12, colors=GREY_800)
    
    # Find and mark the best (lowest) FID value
    best_fid_idx = np.argmin(fid_values)
    best_fid_cfg = cfg_values[best_fid_idx]
    best_fid = fid_values[best_fid_idx]
    ax1.plot(best_fid_cfg, best_fid, 'o', color=PRIMARY, markersize=15, 
           markeredgecolor=GREY_800, markeredgewidth=2)
    ax1.annotate(f'Best: {best_fid:.2f}', 
               xy=(best_fid_cfg, best_fid), 
               xytext=(10, -20),
               textcoords='offset points',
               fontsize=12, fontweight='bold',
               color=PRIMARY,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add shaded region for good FID values
    ax1.axhspan(0, 50, alpha=0.2, color=SUCCESS)
    ax1.axhspan(50, 100, alpha=0.2, color=WARNING)
    ax1.axhspan(100, max(fid_values) * 1.1, alpha=0.2, color=ERROR)
    
    # Inception Score plot (higher is better)
    ax2 = fig.add_subplot(gs[0, 1])
    all_axes.append(ax2)
    ax2.set_facecolor('white')
    ax2.plot(cfg_values, is_values, 'o-', color=SUCCESS, linewidth=2.5, markersize=10)
    ax2.set_title('Inception Score (higher is better)', fontsize=16, fontweight='bold', color=GREY_900, pad=10)
    ax2.set_xlabel('CFG Scale', fontsize=14, fontweight='bold', color=GREY_800)
    ax2.set_ylabel('IS', fontsize=14, fontweight='bold', color=GREY_800)
    ax2.grid(True, linestyle='--', alpha=0.3, color=GREY_300)
    ax2.tick_params(axis='both', which='major', labelsize=12, colors=GREY_800)
    
    # Find and mark the best (highest) IS value
    best_is_idx = np.argmax(is_values)
    best_is_cfg = cfg_values[best_is_idx]
    best_is = is_values[best_is_idx]
    ax2.plot(best_is_cfg, best_is, 'o', color=SUCCESS, markersize=15, 
           markeredgecolor=GREY_800, markeredgewidth=2)
    ax2.annotate(f'Best: {best_is:.2f}', 
               xy=(best_is_cfg, best_is), 
               xytext=(10, 10),
               textcoords='offset points',
               fontsize=12, fontweight='bold',
               color=SUCCESS,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # CLIP Score plot (higher is better)
    ax3 = fig.add_subplot(gs[1, 0])
    all_axes.append(ax3)
    ax3.set_facecolor('white')
    ax3.plot(cfg_values, clip_values, 'o-', color=ERROR, linewidth=2.5, markersize=10)
    ax3.set_title('CLIP Score (higher is better)', fontsize=16, fontweight='bold', color=GREY_900, pad=10)
    ax3.set_xlabel('CFG Scale', fontsize=14, fontweight='bold', color=GREY_800)
    ax3.set_ylabel('CLIP Score', fontsize=14, fontweight='bold', color=GREY_800)
    ax3.grid(True, linestyle='--', alpha=0.3, color=GREY_300)
    ax3.tick_params(axis='both', which='major', labelsize=12, colors=GREY_800)
    
    # Find and mark the best (highest) CLIP value
    best_clip_idx = np.argmax(clip_values)
    best_clip_cfg = cfg_values[best_clip_idx]
    best_clip = clip_values[best_clip_idx]
    ax3.plot(best_clip_cfg, best_clip, 'o', color=ERROR, markersize=15, 
           markeredgecolor=GREY_800, markeredgewidth=2)
    ax3.annotate(f'Best: {best_clip:.4f}', 
               xy=(best_clip_cfg, best_clip), 
               xytext=(10, 10),
               textcoords='offset points',
               fontsize=12, fontweight='bold',
               color=ERROR,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # RefCLIP Score plot (higher is better)
    ax4 = fig.add_subplot(gs[1, 1])
    all_axes.append(ax4)
    ax4.set_facecolor('white')
    ax4.plot(cfg_values, refclip_values, 'o-', color=ACCENT, linewidth=2.5, markersize=10)
    ax4.set_title('RefCLIP Score (higher is better)', fontsize=16, fontweight='bold', color=GREY_900, pad=10)
    ax4.set_xlabel('CFG Scale', fontsize=14, fontweight='bold', color=GREY_800)
    ax4.set_ylabel('RefCLIP Score', fontsize=14, fontweight='bold', color=GREY_800)
    ax4.grid(True, linestyle='--', alpha=0.3, color=GREY_300)
    ax4.tick_params(axis='both', which='major', labelsize=12, colors=GREY_800)
    
    # Find and mark the best (highest) RefCLIP value
    best_refclip_idx = np.argmax(refclip_values)
    best_refclip_cfg = cfg_values[best_refclip_idx]
    best_refclip = refclip_values[best_refclip_idx]
    ax4.plot(best_refclip_cfg, best_refclip, 'o', color=ACCENT, markersize=15, 
           markeredgecolor=GREY_800, markeredgewidth=2)
    ax4.annotate(f'Best: {best_refclip:.4f}', 
               xy=(best_refclip_cfg, best_refclip), 
               xytext=(10, 10),
               textcoords='offset points',
               fontsize=12, fontweight='bold',
               color=ACCENT,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Ensure all x-axes have the same limits for perfect alignment
    all_x_ticks = set()
    for ax in all_axes:
        all_x_ticks.update(ax.get_xticks())
    all_x_ticks = sorted(list(all_x_ticks))
    
    for ax in all_axes:
        ax.set_xticks(cfg_values)  # Use the actual CFG values for ticks
    
    # Use constrained_layout instead of tight_layout for better alignment
    fig.set_constrained_layout(True)
    
    # Save figure with high quality
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Metrics plot saved to {output_path}")

def plot_step_metrics(all_metrics, output_path="step_evaluation_results.png"):
    """Plot evaluation metrics for different step values"""
    # Extract metrics
    step_values = sorted(all_metrics.keys())
    fid_values = [all_metrics[step]['fid'] for step in step_values]
    is_values = [all_metrics[step]['is'] for step in step_values]
    clip_values = [all_metrics[step]['clip'] for step in step_values]
    refclip_values = [all_metrics[step]['refclip'] for step in step_values]
    
    # Create figure with GridSpec for better control over subplot alignment
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.25, hspace=0.3)
    
    fig.suptitle('Image Generation Quality Metrics vs. Diffusion Steps', 
                fontsize=20, fontweight='bold', color=GREY_900, y=0.98)
    
    # Create a list to store all axes for later synchronization
    all_axes = []
    
    # FID plot (lower is better)
    ax1 = fig.add_subplot(gs[0, 0])
    all_axes.append(ax1)
    ax1.set_facecolor('white')
    ax1.plot(step_values, fid_values, 'o-', color=PRIMARY, linewidth=2.5, markersize=10)
    ax1.set_title('FID Score (lower is better)', fontsize=16, fontweight='bold', color=GREY_900, pad=10)
    ax1.set_xlabel('Diffusion Steps', fontsize=14, fontweight='bold', color=GREY_800)
    ax1.set_ylabel('FID', fontsize=14, fontweight='bold', color=GREY_800)
    ax1.grid(True, linestyle='--', alpha=0.3, color=GREY_300)
    ax1.tick_params(axis='both', which='major', labelsize=12, colors=GREY_800)
    
    # Find and mark the best (lowest) FID value
    best_fid_idx = np.argmin(fid_values)
    best_fid_step = step_values[best_fid_idx]
    best_fid = fid_values[best_fid_idx]
    ax1.plot(best_fid_step, best_fid, 'o', color=PRIMARY, markersize=15, 
           markeredgecolor=GREY_800, markeredgewidth=2)
    ax1.annotate(f'Best: {best_fid:.2f}', 
               xy=(best_fid_step, best_fid), 
               xytext=(10, -20),
               textcoords='offset points',
               fontsize=12, fontweight='bold',
               color=PRIMARY,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add shaded region for good FID values
    ax1.axhspan(0, 50, alpha=0.2, color=SUCCESS)
    ax1.axhspan(50, 100, alpha=0.2, color=WARNING)
    ax1.axhspan(100, max(fid_values) * 1.1, alpha=0.2, color=ERROR)
    
    # Inception Score plot (higher is better)
    ax2 = fig.add_subplot(gs[0, 1])
    all_axes.append(ax2)
    ax2.set_facecolor('white')
    ax2.plot(step_values, is_values, 'o-', color=SUCCESS, linewidth=2.5, markersize=10)
    ax2.set_title('Inception Score (higher is better)', fontsize=16, fontweight='bold', color=GREY_900, pad=10)
    ax2.set_xlabel('Diffusion Steps', fontsize=14, fontweight='bold', color=GREY_800)
    ax2.set_ylabel('IS', fontsize=14, fontweight='bold', color=GREY_800)
    ax2.grid(True, linestyle='--', alpha=0.3, color=GREY_300)
    ax2.tick_params(axis='both', which='major', labelsize=12, colors=GREY_800)
    
    # Find and mark the best (highest) IS value
    best_is_idx = np.argmax(is_values)
    best_is_step = step_values[best_is_idx]
    best_is = is_values[best_is_idx]
    ax2.plot(best_is_step, best_is, 'o', color=SUCCESS, markersize=15, 
           markeredgecolor=GREY_800, markeredgewidth=2)
    ax2.annotate(f'Best: {best_is:.2f}', 
               xy=(best_is_step, best_is), 
               xytext=(10, 10),
               textcoords='offset points',
               fontsize=12, fontweight='bold',
               color=SUCCESS,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # CLIP Score plot (higher is better)
    ax3 = fig.add_subplot(gs[1, 0])
    all_axes.append(ax3)
    ax3.set_facecolor('white')
    ax3.plot(step_values, clip_values, 'o-', color=ERROR, linewidth=2.5, markersize=10)
    ax3.set_title('CLIP Score (higher is better)', fontsize=16, fontweight='bold', color=GREY_900, pad=10)
    ax3.set_xlabel('Diffusion Steps', fontsize=14, fontweight='bold', color=GREY_800)
    ax3.set_ylabel('CLIP Score', fontsize=14, fontweight='bold', color=GREY_800)
    ax3.grid(True, linestyle='--', alpha=0.3, color=GREY_300)
    ax3.tick_params(axis='both', which='major', labelsize=12, colors=GREY_800)
    
    # Find and mark the best (highest) CLIP value
    best_clip_idx = np.argmax(clip_values)
    best_clip_step = step_values[best_clip_idx]
    best_clip = clip_values[best_clip_idx]
    ax3.plot(best_clip_step, best_clip, 'o', color=ERROR, markersize=15, 
           markeredgecolor=GREY_800, markeredgewidth=2)
    ax3.annotate(f'Best: {best_clip:.4f}', 
               xy=(best_clip_step, best_clip), 
               xytext=(10, 10),
               textcoords='offset points',
               fontsize=12, fontweight='bold',
               color=ERROR,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # RefCLIP Score plot (higher is better)
    ax4 = fig.add_subplot(gs[1, 1])
    all_axes.append(ax4)
    ax4.set_facecolor('white')
    ax4.plot(step_values, refclip_values, 'o-', color=ACCENT, linewidth=2.5, markersize=10)
    ax4.set_title('RefCLIP Score (higher is better)', fontsize=16, fontweight='bold', color=GREY_900, pad=10)
    ax4.set_xlabel('Diffusion Steps', fontsize=14, fontweight='bold', color=GREY_800)
    ax4.set_ylabel('RefCLIP Score', fontsize=14, fontweight='bold', color=GREY_800)
    ax4.grid(True, linestyle='--', alpha=0.3, color=GREY_300)
    ax4.tick_params(axis='both', which='major', labelsize=12, colors=GREY_800)
    
    # Find and mark the best (highest) RefCLIP value
    best_refclip_idx = np.argmax(refclip_values)
    best_refclip_step = step_values[best_refclip_idx]
    best_refclip = refclip_values[best_refclip_idx]
    ax4.plot(best_refclip_step, best_refclip, 'o', color=ACCENT, markersize=15, 
           markeredgecolor=GREY_800, markeredgewidth=2)
    ax4.annotate(f'Best: {best_refclip:.4f}', 
               xy=(best_refclip_step, best_refclip), 
               xytext=(10, 10),
               textcoords='offset points',
               fontsize=12, fontweight='bold',
               color=ACCENT,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Ensure all x-axes have the same limits for perfect alignment
    all_x_ticks = set()
    for ax in all_axes:
        all_x_ticks.update(ax.get_xticks())
    all_x_ticks = sorted(list(all_x_ticks))
    
    for ax in all_axes:
        ax.set_xticks(step_values)  # Use the actual step values for ticks
    
    # Use constrained_layout instead of tight_layout for better alignment
    fig.set_constrained_layout(True)
    
    # Save figure with high quality
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Step metrics plot saved to {output_path}")

def get_folder_path(cfg=7, steps=None):
    """Get the folder path based on whether it's a CFG or step variation"""
    if steps is None:
        # CFG variation
        return f"data/coco/generated_sd1_4_{cfg}"
    else:
        # Step variation
        return f"data/coco/generated_sd1_4_7_steps_{steps}"

def main():
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load CLIP model
    try:
        print("Loading CLIP model...")
        model_name = "ViT-B/32"
        clip_model, clip_preprocess = clip.load(model_name, device=device, download_root=os.path.expanduser("~/.cache/clip"))
        print(f"CLIP model {model_name} loaded successfully")
    except Exception as e:
        print(f"Error loading CLIP model: {str(e)}")
        return
    
    # Get the list of all available image IDs from the first available folder
    print("\nGetting all available image IDs...")
    first_gen_folder = "data/coco/generated_sd1_4_7"  # Use CFG 7 as reference
    
    all_image_ids = get_image_ids_from_folder(first_gen_folder)
    if not all_image_ids:
        print("Error: No valid image IDs found!")
        return
    
    # Take only the first 1000 image IDs
    all_image_ids = sorted(all_image_ids)[:1000]
    print(f"Using first {len(all_image_ids)} image IDs")
    
    # Parameters for multiple runs
    num_runs = 5
    subset_size = 200  # Number of images to use in each run
    
    # Load COCO captions
    try:
        image_id_to_caption = load_coco_captions()
        print(f"Loaded {len(image_id_to_caption)} image captions from COCO annotations")
    except Exception as e:
        print(f"Error loading COCO captions: {str(e)}")
        return
    
    # Evaluate CFG variations
    all_cfg_metrics = []
    cfg_values = [1, 3, 7, 10, 20]
    
    # Evaluate step variations
    all_step_metrics = []
    step_values = [10, 20, 50, 100, 200, 500]
    
    # Perform multiple evaluation runs
    for run in range(num_runs):
        print(f"\nStarting Run {run + 1}/{num_runs}")
        
        # Select consecutive subset of images for this run
        start_idx = (run * subset_size) % len(all_image_ids)
        end_idx = start_idx + subset_size
        if end_idx > len(all_image_ids):
            # Wrap around to the beginning if we reach the end
            target_image_ids = all_image_ids[start_idx:] + all_image_ids[:end_idx - len(all_image_ids)]
        else:
            target_image_ids = all_image_ids[start_idx:end_idx]
        
        print(f"Selected images {start_idx} to {(start_idx + subset_size - 1) % len(all_image_ids)} for evaluation")
        
        # Load reference images for this subset
        print(f"Loading reference images for run {run + 1}...")
        ref_folder = "data/coco/train2014"
        ref_images = load_and_preprocess_images(ref_folder, target_image_ids)
        
        if not ref_images:
            print(f"Error: No reference images found for run {run + 1}!")
            continue
        
        print(f"Loaded {len(ref_images)} reference images")
        
        # Evaluate CFG variations
        cfg_metrics = {}
        for cfg in cfg_values:
            print(f"\nEvaluating CFG {cfg} (Run {run + 1})...")
            folder_path = get_folder_path(cfg)  # Get CFG variation folder
            
            if os.path.exists(folder_path):
                results = evaluate_folder(ref_images, folder_path, target_image_ids, clip_model, clip_preprocess)
                if results:
                    cfg_metrics[cfg] = results
                    print(f"CFG {cfg} Results (Run {run + 1}):")
                    print(f"  FID Score: {results['fid']:.2f}")
                    print(f"  IS Score: {results['is_mean']:.2f} ± {results['is_std']:.2f}")
                    print(f"  CLIP Score: {results['clip_score']:.4f}")
                    print(f"  RefCLIP Score: {results['refclip_score']:.4f}")
                    
                    # Clear GPU memory after each evaluation
                    torch.cuda.empty_cache()
            else:
                print(f"Warning: Folder not found: {folder_path}")
        
        all_cfg_metrics.append(cfg_metrics)
        
        # Evaluate step variations (using CFG=7)
        step_metrics = {}
        for steps in step_values:
            print(f"\nEvaluating Steps {steps} (Run {run + 1})...")
            folder_path = get_folder_path(7, steps)  # Get step variation folder
            
            if os.path.exists(folder_path):
                results = evaluate_folder(ref_images, folder_path, target_image_ids, clip_model, clip_preprocess)
                if results:
                    step_metrics[steps] = results
                    print(f"Steps {steps} Results (Run {run + 1}):")
                    print(f"  FID Score: {results['fid']:.2f}")
                    print(f"  IS Score: {results['is_mean']:.2f} ± {results['is_std']:.2f}")
                    print(f"  CLIP Score: {results['clip_score']:.4f}")
                    print(f"  RefCLIP Score: {results['refclip_score']:.4f}")
                    
                    # Clear GPU memory after each evaluation
                    torch.cuda.empty_cache()
            else:
                print(f"Warning: Folder not found: {folder_path}")
        
        all_step_metrics.append(step_metrics)
    
    # Plot results with confidence regions
    if all_cfg_metrics:
        print("\nPlotting CFG results...")
        plot_metrics(all_cfg_metrics)
    
    if all_step_metrics:
        print("\nPlotting step results...")
        plot_step_metrics(all_step_metrics)
        
        # Save numerical results
        print("\nSaving numerical results...")
        with open("evaluation_results.txt", "w") as f:
            f.write("=== CFG Evaluation ===\n")
            f.write("Run\tCFG\tFID\tIS\tCLIP\tRefCLIP\n")
            for run, metrics in enumerate(all_cfg_metrics):
                for cfg in sorted(metrics.keys()):
                    f.write(f"{run+1}\t{cfg}\t{metrics[cfg]['fid']:.2f}\t"
                           f"{metrics[cfg]['is_mean']:.2f}\t"
                           f"{metrics[cfg]['clip_score']:.4f}\t"
                           f"{metrics[cfg]['refclip_score']:.4f}\n")
            
            f.write("\n=== Step Evaluation ===\n")
            f.write("Run\tSteps\tFID\tIS\tCLIP\tRefCLIP\n")
            for run, metrics in enumerate(all_step_metrics):
                for steps in sorted(metrics.keys()):
                    f.write(f"{run+1}\t{steps}\t{metrics[steps]['fid']:.2f}\t"
                           f"{metrics[steps]['is_mean']:.2f}\t"
                           f"{metrics[steps]['clip_score']:.4f}\t"
                           f"{metrics[steps]['refclip_score']:.4f}\n")
        
        print("Numerical results saved to 'evaluation_results.txt'")
    else:
        print("No results to plot!")

if __name__ == "__main__":
    main() 