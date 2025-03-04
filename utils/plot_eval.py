import os
import torch
import numpy as np
import matplotlib.pyplot as plt
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
    """Plot FID, IS, and CLIP scores with confidence regions from multiple runs"""
    cfg_values = sorted(all_metrics[0].keys())
    num_runs = len(all_metrics)
    
    # Prepare data arrays for each metric
    fid_scores = np.zeros((num_runs, len(cfg_values)))
    is_scores = np.zeros((num_runs, len(cfg_values)))
    is_stds = np.zeros((num_runs, len(cfg_values)))
    clip_scores = np.zeros((num_runs, len(cfg_values)))
    refclip_scores = np.zeros((num_runs, len(cfg_values)))
    
    # Fill arrays with data from all runs
    for run in range(num_runs):
        for i, cfg in enumerate(cfg_values):
            fid_scores[run, i] = all_metrics[run][cfg]['fid']/10.0  # Normalize FID
            is_scores[run, i] = all_metrics[run][cfg]['is_mean']
            is_stds[run, i] = all_metrics[run][cfg]['is_std']
            clip_scores[run, i] = all_metrics[run][cfg]['clip_score']
            refclip_scores[run, i] = all_metrics[run][cfg]['refclip_score']
    
    # Calculate means and standard deviations across runs
    fid_means = np.mean(fid_scores, axis=0)
    fid_stds = np.std(fid_scores, axis=0)
    is_means = np.mean(is_scores, axis=0)
    is_run_stds = np.std(is_scores, axis=0)
    clip_means = np.mean(clip_scores, axis=0)
    clip_stds = np.std(clip_scores, axis=0)
    refclip_means = np.mean(refclip_scores, axis=0)
    refclip_stds = np.std(refclip_scores, axis=0)
    
    # Invert FID scores (higher is better)
    max_fid = np.max(fid_means + fid_stds) + 1  # Add 1 for padding
    fid_means = max_fid - fid_means
    
    # Create square figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax2 = ax1.twinx()
    
    # Plot FID and IS on primary y-axis with confidence regions
    ln1 = ax1.plot(cfg_values, fid_means, 'b-o', label='Inversed Normalized FID ↑', linewidth=2)
    ax1.fill_between(cfg_values, 
                     fid_means - fid_stds, 
                     fid_means + fid_stds, 
                     color='blue', alpha=0.2)
    
    ln2 = ax1.plot(cfg_values, is_means, 'g-o', label='Inception Score ↑', linewidth=2)
    # Combine both sources of variance for IS (run variance and internal variance)
    total_is_std = np.sqrt(is_run_stds**2 + np.mean(is_stds, axis=0)**2)
    ax1.fill_between(cfg_values,
                     is_means - total_is_std,
                     is_means + total_is_std,
                     color='green', alpha=0.2)
    
    # Plot CLIP scores on secondary y-axis with confidence regions
    ln3 = ax2.plot(cfg_values, clip_means, 'r-o', label='CLIP Score ↑', linewidth=2)
    ax2.fill_between(cfg_values,
                     clip_means - clip_stds,
                     clip_means + clip_stds,
                     color='red', alpha=0.2)
    
    ln4 = ax2.plot(cfg_values, refclip_means, 'm-o', label='RefCLIP Score ↑', linewidth=2)
    ax2.fill_between(cfg_values,
                     refclip_means - refclip_stds,
                     refclip_means + refclip_stds,
                     color='magenta', alpha=0.2)
    
    # Customize plot
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xlabel('CFG Value', fontsize=12)
    ax1.set_ylabel('FID and IS Scores', fontsize=12)
    ax2.set_ylabel('RefCLIP and CLIP Scores', fontsize=12)
    plt.title(f'Evaluation Metrics vs CFG Value\n(Mean ± Std over {num_runs} runs)', fontsize=14, pad=20)
    
    # Adjust y-axis limits to make room for legend
    ax1.set_ylim(top=max(np.max(is_means + total_is_std), np.max(fid_means + fid_stds)) * 1.2)
    ax2.set_ylim(top=max(np.max(clip_means + clip_stds), np.max(refclip_means + refclip_stds)) * 1.2)
    
    # Combine legends from both axes
    lns = ln1 + ln2 + ln3 + ln4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper right', bbox_to_anchor=(0.98, 0.98),
              fontsize=10, framealpha=0.9, edgecolor='white')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {output_path}")
    
    # Print statistical summary
    print("\nStatistical Summary:")
    for i, cfg in enumerate(cfg_values):
        print(f"\nCFG {cfg}:")
        print(f"FID: {fid_scores.mean(axis=0)[i]*10:.2f} ± {fid_stds[i]*10:.2f}")
        print(f"IS: {is_means[i]:.2f} ± {total_is_std[i]:.2f}")
        print(f"CLIP: {clip_means[i]:.4f} ± {clip_stds[i]:.4f}")
        print(f"RefCLIP: {refclip_means[i]:.4f} ± {refclip_stds[i]:.4f}")

def plot_step_metrics(all_metrics, output_path="step_evaluation_results.png"):
    """Plot FID, IS, and CLIP scores with confidence regions from multiple runs for different step counts"""
    step_values = sorted(all_metrics[0].keys())
    num_runs = len(all_metrics)
    
    # Prepare data arrays for each metric
    fid_scores = np.zeros((num_runs, len(step_values)))
    is_scores = np.zeros((num_runs, len(step_values)))
    is_stds = np.zeros((num_runs, len(step_values)))
    clip_scores = np.zeros((num_runs, len(step_values)))
    refclip_scores = np.zeros((num_runs, len(step_values)))
    
    # Fill arrays with data from all runs
    for run in range(num_runs):
        for i, step in enumerate(step_values):
            fid_scores[run, i] = all_metrics[run][step]['fid']/10.0  # Normalize FID
            is_scores[run, i] = all_metrics[run][step]['is_mean']
            is_stds[run, i] = all_metrics[run][step]['is_std']
            clip_scores[run, i] = all_metrics[run][step]['clip_score']
            refclip_scores[run, i] = all_metrics[run][step]['refclip_score']
    
    # Calculate means and standard deviations across runs
    fid_means = np.mean(fid_scores, axis=0)
    fid_stds = np.std(fid_scores, axis=0)
    is_means = np.mean(is_scores, axis=0)
    is_run_stds = np.std(is_scores, axis=0)
    clip_means = np.mean(clip_scores, axis=0)
    clip_stds = np.std(clip_scores, axis=0)
    refclip_means = np.mean(refclip_scores, axis=0)
    refclip_stds = np.std(refclip_scores, axis=0)
    
    # Invert FID scores (higher is better)
    max_fid = np.max(fid_means + fid_stds) + 1  # Add 1 for padding
    fid_means = max_fid - fid_means
    
    # Create square figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax2 = ax1.twinx()
    
    # Plot FID and IS on primary y-axis with confidence regions
    ln1 = ax1.plot(step_values, fid_means, 'b-o', label='Inversed Normalized FID ↑', linewidth=2)
    ax1.fill_between(step_values, 
                     fid_means - fid_stds, 
                     fid_means + fid_stds, 
                     color='blue', alpha=0.2)
    
    ln2 = ax1.plot(step_values, is_means, 'g-o', label='Inception Score ↑', linewidth=2)
    # Combine both sources of variance for IS
    total_is_std = np.sqrt(is_run_stds**2 + np.mean(is_stds, axis=0)**2)
    ax1.fill_between(step_values,
                     is_means - total_is_std,
                     is_means + total_is_std,
                     color='green', alpha=0.2)
    
    # Plot CLIP scores on secondary y-axis with confidence regions
    ln3 = ax2.plot(step_values, clip_means, 'r-o', label='CLIP Score ↑', linewidth=2)
    ax2.fill_between(step_values,
                     clip_means - clip_stds,
                     clip_means + clip_stds,
                     color='red', alpha=0.2)
    
    ln4 = ax2.plot(step_values, refclip_means, 'm-o', label='RefCLIP Score ↑', linewidth=2)
    ax2.fill_between(step_values,
                     refclip_means - refclip_stds,
                     refclip_means + refclip_stds,
                     color='magenta', alpha=0.2)
    
    # Customize plot
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xlabel('Number of Steps', fontsize=12)
    ax1.set_ylabel('FID and IS Scores', fontsize=12)
    ax2.set_ylabel('RefCLIP and CLIP Scores', fontsize=12)
    plt.title(f'Evaluation Metrics vs Steps (CFG=7)\n(Mean ± Std over {num_runs} runs)', fontsize=14, pad=20)
    
    # Use logarithmic scale for x-axis
    ax1.set_xscale('log')
    
    # Adjust y-axis limits to make room for legend
    ax1.set_ylim(top=max(np.max(is_means + total_is_std), np.max(fid_means + fid_stds)) * 1.2)
    ax2.set_ylim(top=max(np.max(clip_means + clip_stds), np.max(refclip_means + refclip_stds)) * 1.2)
    
    # Combine legends from both axes
    lns = ln1 + ln2 + ln3 + ln4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper right', bbox_to_anchor=(0.98, 0.98),
              fontsize=10, framealpha=0.9, edgecolor='white')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {output_path}")
    
    # Print statistical summary
    print("\nStatistical Summary (Steps):")
    for i, step in enumerate(step_values):
        print(f"\nSteps {step}:")
        print(f"FID: {fid_scores.mean(axis=0)[i]*10:.2f} ± {fid_stds[i]*10:.2f}")
        print(f"IS: {is_means[i]:.2f} ± {total_is_std[i]:.2f}")
        print(f"CLIP: {clip_means[i]:.4f} ± {clip_stds[i]:.4f}")
        print(f"RefCLIP: {refclip_means[i]:.4f} ± {refclip_stds[i]:.4f}")

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