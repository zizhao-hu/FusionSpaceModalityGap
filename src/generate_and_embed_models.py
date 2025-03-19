import os
import sys
import json
import torch
import clip
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import argparse
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel
import random
import glob

# Add the repository root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

def load_coco_eval_captions(annotation_file, num_captions=200):
    """Load COCO evaluation captions from json file"""
    print(f"Loading COCO captions from {annotation_file}")
    
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
    unique_image_ids = sorted(list(image_id_to_caption.keys()))
    
    # Use random seed 42 for reproducibility
    random.seed(42)
    selected_image_ids = random.sample(unique_image_ids, min(num_captions, len(unique_image_ids)))
    
    # Create a list of (image_id, caption) pairs
    caption_pairs = []
    for image_id in selected_image_ids:
        # Use the first caption for each image
        caption = image_id_to_caption[image_id][0]
        caption_pairs.append((image_id, caption))
    
    print(f"Loaded {len(caption_pairs)} caption pairs")
    return caption_pairs

def load_model(model_path, cfg_scale=7.0, steps=50):
    """Load a stable diffusion model with the specified UNet"""
    print(f"Loading model from {model_path}")
    
    # Load model with offline fallback
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_cache = os.path.join(cache_dir, "models--CompVis--stable-diffusion-v1-4", "snapshots", "39593d5650112b4cc580433f6b0435385882d819")
    
    if not os.path.exists(model_cache):
        print("Model cache not found. Downloading base model...")
        pipeline = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16,
            use_auth_token=True
        )
        # Save the base model components to cache
        if not os.path.exists(model_cache):
            os.makedirs(model_cache, exist_ok=True)
            pipeline.save_pretrained(model_cache)
        del pipeline
        torch.cuda.empty_cache()
    
    try:
        # Load the pipeline from cache
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_cache,
            torch_dtype=torch.float16,
            local_files_only=True
        ).to("cuda")
        
        # Load the UNet from the specified model path
        if not os.path.exists(os.path.join(model_path, "unet")):
            raise RuntimeError(f"UNet not found at {os.path.join(model_path, 'unet')}")
        
        pipeline.unet = UNet2DConditionModel.from_pretrained(
            os.path.join(model_path, "unet"),
            torch_dtype=torch.float16
        ).to("cuda")
        
        # Disable safety checker for consistency
        pipeline.safety_checker = None
        
        return pipeline
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def generate_images(pipeline, caption_pairs, output_dir, cfg_scale=7.0, steps=50, batch_size=4):
    """Generate images using the loaded model and save them to the output directory"""
    print(f"Generating images to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set fixed seed for deterministic generation
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # Check existing images
    existing_images = {
        int(os.path.basename(img_file).split('_')[-1].split('.')[0]): img_file
        for img_file in glob.glob(os.path.join(output_dir, "*.jpg"))
        if os.path.getsize(img_file) > 1024
    }
    
    # Filter out pairs that already have valid images
    pairs_to_generate = [
        (i, image_id, caption) for i, (image_id, caption) in enumerate(caption_pairs)
        if image_id not in existing_images
    ]
    
    if not pairs_to_generate:
        print(f"All {len(caption_pairs)} images already exist in {output_dir}")
        return True
    
    print(f"Need to generate {len(pairs_to_generate)} images out of {len(caption_pairs)}")
    
    # Process in batches
    for batch_start in range(0, len(pairs_to_generate), batch_size):
        batch_end = min(batch_start + batch_size, len(pairs_to_generate))
        batch = pairs_to_generate[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//batch_size + 1}/{(len(pairs_to_generate)-1)//batch_size + 1}")
        
        # Prepare batch data
        batch_indices = [pair[0] for pair in batch]
        batch_image_ids = [pair[1] for pair in batch]
        batch_captions = [pair[2] for pair in batch]
        batch_seeds = [image_id % 10000 for image_id in batch_image_ids]
        
        try:
            # Generate images for the entire batch
            generators = [torch.Generator("cuda").manual_seed(seed) for seed in batch_seeds]
            batch_outputs = pipeline(
                batch_captions,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                generator=generators,
                num_images_per_prompt=1
            ).images
            
            # Save batch results
            for idx, (image_id, image) in enumerate(zip(batch_image_ids, batch_outputs)):
                image_filename = f"COCO_eval_{image_id:012d}.jpg"
                image_path = os.path.join(output_dir, image_filename)
                
                try:
                    image.save(image_path)
                    if os.path.getsize(image_path) < 1024:
                        print(f"Warning: Generated image {image_filename} is too small")
                    else:
                        print(f"Successfully saved image {image_filename}")
                except Exception as e:
                    print(f"Error saving image {image_filename}: {str(e)}")
            
        except Exception as e:
            print(f"Error generating batch: {str(e)}")
        
        # Clear memory after each batch
        torch.cuda.empty_cache()
    
    # Verify results
    valid_images = sum(
        1 for img_file in os.listdir(output_dir)
        if img_file.endswith(('.jpg', '.png')) and os.path.getsize(os.path.join(output_dir, img_file)) > 1024
    )
    
    print(f"Generation complete: {valid_images}/{len(caption_pairs)} valid images in {output_dir}")
    
    return valid_images >= len(caption_pairs)

def extract_clip_embeddings(image_dir, caption_pairs, output_file, batch_size=16):
    """Extract CLIP embeddings for images and their corresponding captions"""
    print(f"Extracting CLIP embeddings for images in {image_dir}")
    
    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "ViT-B/32"
    clip_model, clip_preprocess = clip.load(model_name, device=device, download_root=os.path.expanduser("~/.cache/clip"))
    
    # Create image_id to caption mapping
    image_id_to_caption = {image_id: caption for image_id, caption in caption_pairs}
    
    # Get all image files
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    
    # Filter out invalid images
    valid_image_files = [
        img_file for img_file in image_files
        if os.path.getsize(img_file) > 1024
    ]
    
    if len(valid_image_files) < len(caption_pairs):
        print(f"Warning: Only found {len(valid_image_files)} valid images out of {len(caption_pairs)} expected")
    
    # Create a mapping of image_id to file path
    image_id_to_file = {}
    for img_file in valid_image_files:
        try:
            # Extract image_id from filename (COCO_eval_000000000001.jpg)
            image_id = int(os.path.basename(img_file).split('_')[-1].split('.')[0])
            image_id_to_file[image_id] = img_file
        except (ValueError, IndexError):
            continue
    
    # Create ordered lists of image files and captions
    ordered_image_files = []
    ordered_captions = []
    
    for image_id, caption in caption_pairs:
        if image_id in image_id_to_file:
            ordered_image_files.append(image_id_to_file[image_id])
            ordered_captions.append(caption)
    
    if not ordered_image_files:
        print(f"No valid images found in {image_dir}")
        return False
    
    print(f"Processing {len(ordered_image_files)} image-caption pairs")
    
    # Process in batches
    all_image_features = []
    all_text_features = []
    
    for i in range(0, len(ordered_image_files), batch_size):
        end_idx = min(i + batch_size, len(ordered_image_files))
        batch_image_files = ordered_image_files[i:end_idx]
        batch_captions = ordered_captions[i:end_idx]
        
        # Load and preprocess images
        batch_images = []
        for img_file in batch_image_files:
            try:
                img = Image.open(img_file).convert("RGB")
                batch_images.append(img)
            except Exception as e:
                print(f"Error loading image {img_file}: {str(e)}")
                # Use a blank image as a placeholder
                batch_images.append(Image.new("RGB", (512, 512), color="black"))
        
        # Preprocess images and text
        processed_images = torch.stack([clip_preprocess(img) for img in batch_images]).to(device)
        text_tokens = clip.tokenize(batch_captions).to(device)
        
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
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.savez(
        output_file,
        image_embeddings=all_image_features,
        text_embeddings=all_text_features,
        image_files=ordered_image_files,
        captions=ordered_captions
    )
    
    print(f"Saved embeddings to {output_file}")
    return True

def process_model_generation(gen_number, model_type, caption_pairs, cfg_scale=7.0, steps=50):
    """Process a specific model generation and type"""
    # Define model and output directories
    if model_type == "regular":
        model_dir = f"models/sd_to_sd_cfg_{int(cfg_scale)}_steps_{steps}_gen_{gen_number}"
        output_dir = f"data/coco/gen_large/sd_to_sd_cfg_{int(cfg_scale)}_steps_{steps}_gen_{gen_number}"
    elif model_type == "finetune":
        model_dir = f"models/sd_to_sd_cfg_{int(cfg_scale)}_steps_{steps}_gen_{gen_number}_finetune"
        output_dir = f"data/coco/gen_large/sd_to_sd_cfg_{int(cfg_scale)}_steps_{steps}_gen_{gen_number}_finetune"
    elif model_type == "baseline":
        model_dir = f"models/sd_to_sd_cfg_{int(cfg_scale)}_steps_{steps}_gen_{gen_number}_baseline"
        output_dir = f"data/coco/gen_large/sd_to_sd_cfg_{int(cfg_scale)}_steps_{steps}_gen_{gen_number}_baseline"
    else:
        print(f"Unknown model type: {model_type}")
        return False
    
    # Check if model exists
    if not os.path.exists(os.path.join(model_dir, "unet")):
        print(f"Model not found at {model_dir}")
        return False
    
    # Define embedding output file
    embedding_dir = "data/embeddings"
    embedding_file = os.path.join(embedding_dir, f"CLIP_openai_clip-vit-base-patch32_embeddings_gen_{gen_number}{'' if model_type == 'regular' else '_' + model_type}.npz")
    
    # Check if embeddings already exist
    if os.path.exists(embedding_file):
        print(f"Embeddings already exist at {embedding_file}")
        return True
    
    # Load model
    pipeline = load_model(model_dir, cfg_scale, steps)
    if pipeline is None:
        return False
    
    # Generate images
    success = generate_images(pipeline, caption_pairs, output_dir, cfg_scale, steps)
    
    # Clean up pipeline to free memory
    del pipeline
    torch.cuda.empty_cache()
    
    if not success:
        print(f"Failed to generate all images for {model_dir}")
        return False
    
    # Extract CLIP embeddings
    return extract_clip_embeddings(output_dir, caption_pairs, embedding_file)

def main():
    parser = argparse.ArgumentParser(description='Generate images and extract CLIP embeddings for specific model generations')
    parser.add_argument('--cfg_scale', type=float, default=7.0, help='CFG scale for image generation')
    parser.add_argument('--steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--num_captions', type=int, default=200, help='Number of captions to use')
    parser.add_argument('--force_generate', action='store_true', help='Force generation of images even if they already exist')
    
    args = parser.parse_args()
    
    # Define target generations and model types
    target_generations = [1, 2, 5, 9, 10]
    model_types = ["regular", "finetune", "baseline"]
    
    # Load COCO evaluation captions
    annotation_file = "data/coco/annotations/captions_val2014.json"
    caption_pairs = load_coco_eval_captions(annotation_file, args.num_captions)
    
    if not caption_pairs:
        print("Failed to load caption pairs")
        return
    
    # Process each generation and model type
    for gen_number in target_generations:
        for model_type in model_types:
            print(f"\n{'='*80}")
            print(f"Processing Generation {gen_number}, Type: {model_type}")
            print(f"{'='*80}")
            
            success = process_model_generation(gen_number, model_type, caption_pairs, args.cfg_scale, args.steps)
            
            if success:
                print(f"Successfully processed Generation {gen_number}, Type: {model_type}")
            else:
                print(f"Failed to process Generation {gen_number}, Type: {model_type}")
    
    print("\nAll processing complete!")

if __name__ == "__main__":
    main() 