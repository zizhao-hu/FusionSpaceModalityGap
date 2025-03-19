import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import random
import json
import cv2
from skimage.color import rgb2gray
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, InterpolationMode, Normalize
from tqdm.auto import tqdm
import argparse

# Import color constants
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from utils.colors import *

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

def calculate_fid_and_clip(image_files, device="cuda"):
    """Calculate FID, IS, and CLIP-based metrics for the given images using the images as their own reference."""
    print(f"Calculating FID, IS, and CLIP metrics for {len(image_files)} images")
    
    # Define transforms for FID calculation
    transform = Compose([
        Resize(299, interpolation=InterpolationMode.BILINEAR),
        CenterCrop(299),
        ToTensor(),
    ])
    
    # Load images
    images = []
    pil_images = []  # Store original PIL images for CLIP
    
    for img_path in tqdm(image_files, desc="Loading images for FID/IS"):
        try:
            img = Image.open(img_path).convert('RGB')
            pil_images.append(img)
            
            img_tensor = transform(img)
            # Scale to [0, 255] and convert to uint8 for FID
            img_tensor = (img_tensor * 255).to(torch.uint8)
            images.append(img_tensor)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    
    if not images:
        print("No valid images loaded")
        return None
    
    # Convert to tensor
    image_stack = torch.stack(images).to(device)
    
    # Initialize scores dictionary
    scores = {'fid': None, 'is_mean': None, 'is_std': None, 'clip_score': None, 'rmg': None, 'l2m': None, 'clip_variance': None}
    
    # Calculate FID using half the images as reference, half as generated
    try:
        mid_point = len(images) // 2
        ref_stack = image_stack[:mid_point]
        gen_stack = image_stack[mid_point:]
        
        fid = FrechetInceptionDistance(normalize=True).to(device)
        
        # Update FID with reference images
        fid.update(ref_stack, real=True)
        
        # Update FID with "generated" images
        fid.update(gen_stack, real=False)
        
        scores['fid'] = float(fid.compute())
        
        del fid
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error calculating FID: {e}")
    
    # Calculate Inception Score
    try:
        inception_score = InceptionScore(normalize=True).to(device)
        inception_score.update(image_stack)
        is_mean, is_std = inception_score.compute()
        scores['is_mean'] = float(is_mean)
        scores['is_std'] = float(is_std)
        
        del inception_score
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error calculating IS: {e}")
    
    # Calculate CLIP metrics
    try:
        print("Loading CLIP model...")
        model_name = "ViT-B/32"
        clip_model, clip_preprocess = clip.load(model_name, device=device, download_root=os.path.expanduser("~/.cache/clip"))
        print(f"CLIP model {model_name} loaded successfully")
        
        # Calculate CLIP scores and additional metrics
        clip_scores = []
        
        # Process in smaller batches
        clip_batch_size = 16
        
        # Store image and text features for additional metrics
        all_image_features = []
        all_text_features = []
        
        for i in range(0, len(pil_images), clip_batch_size):
            end_idx = min(i + clip_batch_size, len(pil_images))
            clip_batch = pil_images[i:end_idx]
            
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
        
        # Save the embeddings
        output_dir = os.path.join("data", "embeddings")
        os.makedirs(output_dir, exist_ok=True)
        embedding_file = os.path.join(output_dir, "CLIP_coco_val_embeddings_real.npz")
        np.savez(embedding_file, 
                 image_embeddings=all_image_features, 
                 text_embeddings=all_text_features, 
                 captions=captions, 
                 file_names=[os.path.basename(p) for p in image_files[:len(all_image_features)]])
        print(f"Saved embeddings to {embedding_file}")
        
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error calculating CLIP metrics: {e}")
    
    return scores

def main():
    parser = argparse.ArgumentParser(description='Calculate metrics for COCO validation images')
    parser.add_argument('--coco_dir', type=str, default=os.path.join("data", "coco", "val2014"), 
                        help='Directory containing COCO validation images')
    parser.add_argument('--output_dir', type=str, default=os.path.join("vis", "t2i", "metrics"), 
                        help='Output directory for results')
    parser.add_argument('--num_images', type=int, default=200, 
                        help='Number of images to analyze')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for image selection')
    parser.add_argument('--use_default_values', action='store_true',
                        help='Use default real metric values instead of calculating them')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # If using default values, skip the calculation and use the provided metrics
    if args.use_default_values:
        print("Using default real metric values...")
        default_metrics = {
            'saturation': 40.07,
            'color_std': 62.61,
            'contrast': 0.5766,
            'brightness': 114.27,
            'colorfulness': 41.03,
            'fid': 220.4891,
            'is_mean': 6.5722,
            'is_std': 0.4394,
            'clip_score': 0.6099,
            'rmg': 0.7482,
            'l2m': 12.0547,
            'clip_variance': 0.1113
        }
        
        # Save the default metrics to CSV
        update_metrics_files(args.output_dir, default_metrics)
        return
    
    # Get all image files in the COCO validation directory
    image_files = sorted(glob.glob(os.path.join(args.coco_dir, "*.jpg")))
    if not image_files:
        print(f"No images found in {args.coco_dir}")
        return
    
    print(f"Found {len(image_files)} images in {args.coco_dir}")
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Randomly select the specified number of images
    if len(image_files) > args.num_images:
        image_files = random.sample(image_files, args.num_images)
    
    print(f"Selected {len(image_files)} images for analysis")
    
    # Analyze color distribution for each image
    color_results = []
    for img_path in tqdm(image_files, desc="Analyzing color distribution"):
        color_data = analyze_color_distribution(img_path)
        if color_data:
            color_results.append(color_data)
    
    # Calculate FID, IS, and CLIP metrics
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    scores = calculate_fid_and_clip(image_files, device=device)
    
    # Calculate mean values for color metrics
    mean_saturation = np.mean([r['saturation'] for r in color_results])
    mean_color_std = np.mean([r['color_std'] for r in color_results])
    mean_contrast = np.mean([r['contrast'] for r in color_results])
    mean_brightness = np.mean([r['brightness'] for r in color_results])
    mean_colorfulness = np.mean([r['colorfulness'] for r in color_results])
    
    # Print the results
    print("\n===== COCO Validation Image Metrics =====")
    print(f"Saturation: {mean_saturation:.2f}")
    print(f"Color Std: {mean_color_std:.2f}")
    print(f"Contrast: {mean_contrast:.4f}")
    print(f"Brightness: {mean_brightness:.2f}")
    print(f"Colorfulness: {mean_colorfulness:.2f}")
    
    if scores:
        print(f"FID: {scores['fid']:.4f}")
        print(f"IS: {scores['is_mean']:.4f} ± {scores['is_std']:.4f}")
        print(f"CLIP Score: {scores['clip_score']:.4f}")
        print(f"RMG: {scores['rmg']:.4f}")
        print(f"L2M: {scores['l2m']:.4f}")
        print(f"CLIP Variance: {scores['clip_variance']:.4f}")
    
    # Combine all metrics into a single dictionary for output
    all_metrics = {
        'saturation': mean_saturation,
        'color_std': mean_color_std,
        'contrast': mean_contrast,
        'brightness': mean_brightness,
        'colorfulness': mean_colorfulness
    }
    
    if scores:
        all_metrics.update({
            'fid': scores['fid'],
            'is_mean': scores['is_mean'],
            'is_std': scores['is_std'],
            'clip_score': scores['clip_score'],
            'rmg': scores['rmg'],
            'l2m': scores['l2m'],
            'clip_variance': scores['clip_variance']
        })
    
    # Save the metrics to various CSV files
    update_metrics_files(args.output_dir, all_metrics)
    
    # Save detailed color metrics for each image
    detailed_csv_path = os.path.join(args.output_dir, "coco_val_detailed_metrics.csv")
    with open(detailed_csv_path, 'w') as f:
        f.write("Image,Saturation,Color_Std,Contrast,Brightness,Colorfulness\n")
        for r in color_results:
            img_name = os.path.basename(r['img_path'])
            f.write(f"{img_name},{r['saturation']:.2f},{r['color_std']:.2f},{r['contrast']:.4f},{r['brightness']:.2f},{r['colorfulness']:.2f}\n")
    
    print(f"Detailed metrics saved to {detailed_csv_path}")

def update_metrics_files(output_dir, metrics):
    """Save metrics to CSV files in the appropriate format"""
    # Save the results to a CSV file
    csv_path = os.path.join(output_dir, "coco_val_metrics.csv")
    with open(csv_path, 'w') as f:
        f.write("Metric,Value\n")
        for key, value in metrics.items():
            f.write(f"{key},{value:.4f}\n")
    
    print(f"Results saved to {csv_path}")
    
    # Update the metrics_results.csv file with Real image metrics in the correct format
    metrics_results_path = os.path.join(output_dir, "metrics_results.csv")
    
    # Create or update the file
    try:
        # Check if file exists
        if os.path.exists(metrics_results_path):
            # Read existing content
            with open(metrics_results_path, 'r') as f:
                lines = f.readlines()
            
            # Find if there's a line with Real group
            real_line_idx = None
            for i, line in enumerate(lines):
                if ",Real," in line:
                    real_line_idx = i
                    break
            
            # Update or append Real line
            real_line = f"0,Real,0,{metrics.get('saturation', 0):.4f},0.0000,{metrics.get('contrast', 0):.4f},0.0000,{metrics.get('brightness', 0):.4f},0.0000,{metrics.get('color_std', 0):.4f},0.0000,{metrics.get('fid', 0):.4f},{metrics.get('is_mean', 0):.4f},{metrics.get('clip_score', 0):.4f},{metrics.get('rmg', 0):.4f},{metrics.get('l2m', 0):.4f},{metrics.get('clip_variance', 0):.4f}\n"
            
            if real_line_idx is not None:
                lines[real_line_idx] = real_line
            else:
                # If header doesn't exist, add it
                if not lines or not lines[0].startswith("Generation,Group,"):
                    lines.insert(0, "Generation,Group,Batch,Saturation,Saturation_Std,Contrast,Contrast_Std,Brightness,Brightness_Std,Color_Std,Color_Std_Std,FID,IS,CLIP_Score,RMG,L2M,CLIP_Variance\n")
                lines.append(real_line)
            
            with open(metrics_results_path, 'w') as f:
                f.writelines(lines)
        else:
            # Create new file with header and Real line
            with open(metrics_results_path, 'w') as f:
                f.write("Generation,Group,Batch,Saturation,Saturation_Std,Contrast,Contrast_Std,Brightness,Brightness_Std,Color_Std,Color_Std_Std,FID,IS,CLIP_Score,RMG,L2M,CLIP_Variance\n")
                f.write(f"0,Real,0,{metrics.get('saturation', 0):.4f},0.0000,{metrics.get('contrast', 0):.4f},0.0000,{metrics.get('brightness', 0):.4f},0.0000,{metrics.get('color_std', 0):.4f},0.0000,{metrics.get('fid', 0):.4f},{metrics.get('is_mean', 0):.4f},{metrics.get('clip_score', 0):.4f},{metrics.get('rmg', 0):.4f},{metrics.get('l2m', 0):.4f},{metrics.get('clip_variance', 0):.4f}\n")
        
        print(f"Updated Real image metrics in {metrics_results_path}")
    except Exception as e:
        print(f"Error updating metrics_results.csv: {e}")

if __name__ == "__main__":
    main() 