import os
import numpy as np
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, InterpolationMode
from PIL import Image
import glob
from tqdm.auto import tqdm

def calculate_metrics_for_batch(gen_stack, ref_stack, clip_model=None, clip_preprocess=None, 
                              batch_size=32, device="cuda", gen=None):
    """Calculate FID, IS, and CLIP scores for a batch of images.
    
    Args:
        gen_stack: Stack of generated images
        ref_stack: Stack of reference images
        clip_model: CLIP model for CLIP score calculation
        clip_preprocess: CLIP preprocessing function
        batch_size: Batch size for processing
        device: Device to use for calculations
        gen: Generation number for logging
        
    Returns:
        Dictionary containing calculated metrics
    """
    scores = {'fid': None, 'is_mean': None, 'is_std': None, 'clip_score': None, 'rmg': None, 'l2m': None, 'clip_variance': None}
    
    # Calculate FID
    try:
        fid = FrechetInceptionDistance(normalize=True).to(device)
        
        # Update FID with reference images in batches
        for i in range(0, len(ref_stack), batch_size):
            batch = ref_stack[i:i+batch_size].to(device)
            fid.update(batch, real=True)
        
        # Update FID with generated images in batches
        for i in range(0, len(gen_stack), batch_size):
            batch = gen_stack[i:i+batch_size].to(device)
            fid.update(batch, real=False)
        
        scores['fid'] = float(fid.compute())
        del fid
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error calculating FID: {e}")
    
    # Calculate Inception Score
    try:
        inception_score = InceptionScore(normalize=True).to(device)
        # Process in batches
        for i in range(0, len(gen_stack), batch_size):
            batch = gen_stack[i:i+batch_size].to(device)
            inception_score.update(batch)
        
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
            # Process in smaller batches for CLIP
            clip_batch_size = 16
            
            # Store image and text features for additional metrics
            all_image_features = []
            all_text_features = []
            clip_scores = []
            
            for i in range(0, len(gen_stack), clip_batch_size):
                end_idx = min(i + clip_batch_size, len(gen_stack))
                batch_images = gen_stack[i:end_idx]
                
                # Create generic captions for the batch
                batch_captions = ["A photograph"] * (end_idx - i)
                
                # Convert tensors to PIL Images for CLIP preprocessing
                pil_images = []
                for img_tensor in batch_images:
                    # Convert from [0, 255] to [0, 1]
                    img_tensor = img_tensor.float() / 255.0
                    # Convert to PIL Image
                    img = Image.fromarray((img_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                    pil_images.append(img)
                
                # Preprocess images and text
                processed_images = torch.stack([clip_preprocess(img) for img in pil_images]).to(device)
                text_tokens = clip.tokenize(batch_captions).to(device)
                
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
                """Calculate cosine dissimilarity between corresponding rows of A and B"""
                numerator = np.einsum('ij,ij->i', A, B)
                normA = np.linalg.norm(A, axis=1)
                normB = np.linalg.norm(B, axis=1)
                cos_sim = numerator / (normA * normB)
                return 1.0 - cos_sim  # dissimilarity
            
            def sum_pairwise_cos_dissim(M):
                """Calculate sum of pairwise cosine dissimilarities within matrix M"""
                dot_mat = M @ M.T
                norms = np.linalg.norm(M, axis=1)
                norm_mat = np.outer(norms, norms)
                cos_sim_mat = dot_mat / norm_mat
                cos_dissim_mat = 1.0 - cos_sim_mat
                # Zero out the diagonal (self-dissimilarity)
                np.fill_diagonal(cos_dissim_mat, 0.0)
                return np.sum(cos_dissim_mat)
            
            # Calculate RMG using the formula from plot_gap.py
            N = all_image_features.shape[0]
            row_dissim_xy = cosine_dissim_rowwise(all_text_features, all_image_features)
            numerator = np.mean(row_dissim_xy)
            
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
            import traceback
            traceback.print_exc()
    
    return scores

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
        if gen == 10:
            np.random.seed(42)
    else:
        # Use different seed each time for other generations
        np.random.seed(None)
    
    # Randomly select batch_size images
    selected_indices = np.random.choice(len(image_files), size=batch_size, replace=False)
    selected_files = [image_files[i] for i in selected_indices]
    
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
    
    # Calculate metrics using the consolidated function
    scores = calculate_metrics_for_batch(gen_stack, ref_stack, clip_model, clip_preprocess, 
                                       batch_size=32, device=device, gen=gen)
    
    return scores

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
    
    # Calculate metrics using the consolidated function
    scores = calculate_metrics_for_batch(gen_stack, ref_stack, clip_model, clip_preprocess, 
                                       batch_size=32, device=device)
    
    return scores

def load_reference_images(ref_folder, max_images=1000, device="cuda"):
    """Load reference images for FID calculation
    
    Args:
        ref_folder: Path to folder containing reference images
        max_images: Maximum number of images to load
        device: Device to use for tensor operations
        
    Returns:
        Tensor stack of reference images or None if loading fails
    """
    try:
        ref_files = sorted(glob.glob(os.path.join(ref_folder, "*.jpg")))
        if not ref_files:
            raise ValueError(f"No .jpg images found in {ref_folder}")
            
        if len(ref_files) > max_images:
            print(f"Limiting reference images to {max_images} from {len(ref_files)} found")
            ref_files = ref_files[:max_images]
        
        transform = Compose([
            Resize(299, interpolation=InterpolationMode.BILINEAR),
            CenterCrop(299),
            ToTensor(),
        ])
        
        ref_images = []
        failed_images = []
        
        for img_path in tqdm(ref_files, desc="Loading reference images"):
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)
                # Scale to [0, 255] and convert to uint8 for FID
                img_tensor = (img_tensor * 255).to(torch.uint8)
                ref_images.append(img_tensor)
            except Exception as e:
                failed_images.append((img_path, str(e)))
                continue
        
        if not ref_images:
            raise ValueError(f"No valid reference images loaded from {ref_folder}")
        
        if failed_images:
            print(f"\nFailed to load {len(failed_images)} reference images:")
            for img_path, error in failed_images:
                print(f"- {img_path}: {error}")
        
        ref_stack = torch.stack(ref_images).to(device)
        print(f"\nSuccessfully loaded {len(ref_images)} reference images")
        return ref_stack
        
    except Exception as e:
        print(f"Error loading reference images: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_clip_model(device="cuda"):
    """Load CLIP model and preprocessor
    
    Args:
        device: Device to use for model operations
        
    Returns:
        Tuple of (clip_model, clip_preprocess) or (None, None) if loading fails
    """
    try:
        print(f"Loading CLIP model on {device}...")
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        print("Successfully loaded CLIP model")
        return clip_model, clip_preprocess
    except Exception as e:
        print(f"Error loading CLIP model: {e}")
        import traceback
        traceback.print_exc()
        return None, None 