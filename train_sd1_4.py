import os
import os
import json
from datasets import load_dataset, Dataset
from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.optimization import get_scheduler
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
import evaluate
import numpy as np
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import shutil

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

def create_dataset(image_dir, annotation_file):
    """Create dataset with images and their corresponding captions"""
    image_id_to_caption = load_coco_captions(annotation_file)
    
    # Get all image files and their IDs
    image_files = []
    captions = []
    
    for img_file in os.listdir(image_dir):
        if img_file.endswith('.jpg') or img_file.endswith('.png'):
            # Extract image_id from filename (assuming format like "COCO_train2014_000000000001.jpg")
            image_id = int(img_file.split('_')[-1].split('.')[0])
            
            if image_id in image_id_to_caption:
                image_files.append(os.path.join(image_dir, img_file))
                # Use the first caption for this image
                captions.append(image_id_to_caption[image_id][0])
    
    return Dataset.from_dict({
        "image": image_files,
        "caption": captions
    })

def generate_gen_zero_images(cfg_scale, num_images=100, steps=50):
    """Generate gen_0 images using the pretrained model with consistent naming convention"""
    print(f"\nGenerating Generation 0 images with pretrained model using CFG scale {cfg_scale}, {num_images} images and {steps} steps")
    
    # Set output directory for generated images with consistent naming - include steps
    output_dir = f"data/coco/sd_to_sd_cfg_{int(cfg_scale)}_steps_{steps}_gen_0"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set fixed seed for deterministic generation
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    try:
        # Load COCO captions
        coco_captions_file = "data/coco/annotations/captions_train2014.json"
        if not os.path.exists(coco_captions_file):
            raise RuntimeError(f"COCO captions file not found: {coco_captions_file}")
        
        with open(coco_captions_file, 'r') as f:
            annotations = json.load(f)
        
        # Create a dictionary to store the first caption for each image ID
        image_id_to_first_caption = {}
        count = 0
        for ann in annotations['annotations']:
            image_id = ann['image_id']
            if image_id not in image_id_to_first_caption and count < num_images:  # Only take up to num_images
                image_id_to_first_caption[image_id] = ann['caption']
                count += 1
            if count >= num_images:
                break
        
        # Convert to list of tuples and sort by image_id
        image_id_caption_pairs = sorted(list(image_id_to_first_caption.items()))
        
        # Take only the first num_images items
        selected_pairs = image_id_caption_pairs[:num_images]
        
        # Extract image IDs and captions
        image_ids = [pair[0] for pair in selected_pairs]
        captions = [pair[1] for pair in selected_pairs]
        
        # Get a more accurate list of existing VALID images by checking file sizes
        existing_valid_images = {}
        if os.path.exists(output_dir):
            for img_file in os.listdir(output_dir):
                if img_file.endswith(('.jpg', '.png')):
                    try:
                        image_id = int(img_file.split('_')[-1].split('.')[0])
                        file_path = os.path.join(output_dir, img_file)
                        # Only consider files larger than 1KB as valid images
                        if os.path.getsize(file_path) > 1024:
                            existing_valid_images[image_id] = file_path
                    except (ValueError, IndexError):
                        continue
        
        # Filter out pairs that already have valid images
        pairs_to_generate = [(image_id, caption) for image_id, caption in selected_pairs 
                           if image_id not in existing_valid_images]
        
        valid_count = len(existing_valid_images)
        print(f"Found {valid_count} valid existing images out of {num_images} needed")
        
        if not pairs_to_generate:
            print(f"All {num_images} images already exist and are valid in {output_dir}")
            return output_dir
        
        print(f"Need to generate {len(pairs_to_generate)} images to reach {num_images} total")
        
        # Load the pretrained model
        print("Loading Stable Diffusion model (downloading if needed)...")
        model_id = "CompVis/stable-diffusion-v1-4"
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_auth_token=True  # Will use Hugging Face token if set
        ).to("cuda")
        
        # Set deterministic generation
        pipeline.set_progress_bar_config(disable=None)
        pipeline.safety_checker = None  # Disable safety checker for consistency
        
        # Process in batches
        batch_size = 4  # Adjust based on GPU memory
        for batch_start in range(0, len(pairs_to_generate), batch_size):
            batch_end = min(batch_start + batch_size, len(pairs_to_generate))
            batch = pairs_to_generate[batch_start:batch_end]
            
            print(f"\nProcessing batch {batch_start//batch_size + 1}/{(len(pairs_to_generate)-1)//batch_size + 1}")
            
            # Prepare batch data
            batch_image_ids = [pair[0] for pair in batch]
            batch_captions = [pair[1] for pair in batch]
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
                    image_filename = f"COCO_train2014_{image_id:012d}.jpg"
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
        
        # Clean up
        del pipeline
        torch.cuda.empty_cache()
        
        # Verify results
        valid_images = sum(
            1 for img_file in os.listdir(output_dir)
            if img_file.endswith(('.jpg', '.png')) and os.path.getsize(os.path.join(output_dir, img_file)) > 1024
        )
        
        print(f"\nGeneration complete: {valid_images}/{num_images} valid images in {output_dir}")
        
        if valid_images < num_images:
            print(f"Warning: Only generated {valid_images}/{num_images} valid images")
        
        return output_dir
    
    except Exception as e:
        print(f"Error generating images: {str(e)}")
        return None

def check_and_create_gen_zero(cfg_scale, num_images=100, steps=50):
    """Check if Generation 0 images exist and create them if needed"""
    # Set output directory for generation 0 images - include steps
    output_dir = f"data/coco/sd_to_sd_cfg_{int(cfg_scale)}_steps_{steps}_gen_0"
    
    # Check if directory exists
    if not os.path.exists(output_dir):
        print(f"Generation 0 directory {output_dir} does not exist. Creating it...")
        os.makedirs(output_dir, exist_ok=True)
    
    # Count valid images in the directory
    valid_images = 0
    if os.path.exists(output_dir):
        for img_file in os.listdir(output_dir):
            if img_file.endswith(('.jpg', '.png')):
                file_path = os.path.join(output_dir, img_file)
                # Only count files larger than 1KB as valid images
                if os.path.getsize(file_path) > 1024:
                    valid_images += 1
    
    print(f"Found {valid_images} valid Generation 0 images out of {num_images} needed")
    
    # If we have enough valid images, return success
    if valid_images >= num_images:
        print(f"Generation 0 is complete with {valid_images} valid images")
        return True
    
    # Otherwise, generate the images
    print(f"Need to generate Generation 0 images. Calling generate_gen_zero_images...")
    generate_gen_zero_images(cfg_scale, num_images, steps)
    
    # Verify again after generation
    valid_images = 0
    if os.path.exists(output_dir):
        for img_file in os.listdir(output_dir):
            if img_file.endswith(('.jpg', '.png')):
                file_path = os.path.join(output_dir, img_file)
                if os.path.getsize(file_path) > 1024:
                    valid_images += 1
    
    # Return success if we have enough valid images now
    success = valid_images >= num_images
    if success:
        print(f"Successfully created Generation 0 with {valid_images} valid images")
    else:
        print(f"WARNING: Generation 0 is still incomplete with only {valid_images}/{num_images} valid images")
    
    return success

def check_training_data(data_dir, required_count=100):
    """Check if training data directory has required number of valid images"""
    if not os.path.exists(data_dir):
        print(f"Training data directory not found: {data_dir}")
        return False
        
    # Count valid images (larger than 1KB)
    valid_images = sum(
        1 for img_file in os.listdir(data_dir)
        if (img_file.endswith(('.jpg', '.png')) and 
            os.path.getsize(os.path.join(data_dir, img_file)) > 1024)
    )
    
    print(f"Found {valid_images} valid images in {data_dir}")
    
    if valid_images < required_count:
        print(f"Insufficient images. Need {required_count}, found {valid_images}")
        return False
        
    return True

def generate_missing_images(data_dir, required_count=100):
    """Generate missing images using base SD1.4 model"""
    print(f"Generating missing images in {data_dir}")
    
    # Initialize SD pipeline
    model_id = "CompVis/stable-diffusion-v1-4"
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    ).to("cuda")
    
    # Generate images until we reach required count
    existing_count = len([f for f in os.listdir(data_dir) 
                         if f.endswith(('.jpg', '.png'))])
    
    for i in range(existing_count, required_count):
        prompt = f"A high quality photograph {i}"  # You might want to customize prompts
        image = pipeline(prompt).images[0]
        
        # Save image
        image_path = os.path.join(data_dir, f"generated_{i:04d}.jpg")
        image.save(image_path)
        print(f"Generated image {i+1}/{required_count}")
        
    del pipeline
    torch.cuda.empty_cache()

def train_stable_diffusion(cfg_scale, gen_number, input_model_path=None, steps=50, epochs=5):
    """Train stable diffusion model with specified parameters"""
    try:
        model_id = "CompVis/stable-diffusion-v1-4"
        train_data_dir = f"data/coco/sd_to_sd_cfg_{int(cfg_scale)}_steps_{steps}_gen_{gen_number-1}"
        output_dir = f"models/sd_to_sd_cfg_{int(cfg_scale)}_steps_{steps}_gen_{gen_number}"
        
        num_images = 100
        batch_size = 1
        max_train_steps = (num_images // batch_size) * epochs
        save_steps = max(max_train_steps // 10, 1)
        
        print(f"\nTraining model for CFG {cfg_scale}, Gen {gen_number}, {epochs} epochs, {steps} steps")
        os.makedirs(output_dir, exist_ok=True)

        # Load models with offline fallback
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_cache = os.path.join(cache_dir, "models--CompVis--stable-diffusion-v1-4", "snapshots", "39593d5650112b4cc580433f6b0435385882d819")
        
        # For generation 1, we need to download the base model if it's not in cache
        if gen_number == 1:
            print("Generation 1: Downloading base model if needed...")
            try:
                pipeline = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    use_auth_token=True
                )
                # Save the base model components to cache
                if not os.path.exists(model_cache):
                    os.makedirs(model_cache, exist_ok=True)
                    pipeline.save_pretrained(model_cache)
                del pipeline
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error downloading base model: {str(e)}")
                if not os.path.exists(model_cache):
                    raise RuntimeError("Failed to download base model and no cached model found")

        print(f"Loading models from: {model_cache if gen_number == 1 else input_model_path}")
        try:
            # Load tokenizer and text encoder from base model
            tokenizer = CLIPTokenizer.from_pretrained(model_cache, subfolder="tokenizer", local_files_only=True)
            text_encoder = CLIPTextModel.from_pretrained(model_cache, subfolder="text_encoder", local_files_only=True).to("cuda")
            
            # Load UNet based on generation number
            if gen_number == 1:
                print("Loading base UNet model for Generation 1...")
                unet = UNet2DConditionModel.from_pretrained(
                    os.path.join(model_cache, "unet"),
                    local_files_only=True
                ).to("cuda")
            else:
                if not input_model_path or not os.path.exists(f"{input_model_path}/unet"):
                    raise RuntimeError(f"Previous generation model not found at {input_model_path}")
                print(f"Loading UNet from previous generation: {input_model_path}")
                unet = UNet2DConditionModel.from_pretrained(
                    f"{input_model_path}/unet"
                ).to("cuda")
            
            # Load scheduler from base model
            noise_scheduler = DDPMScheduler.from_pretrained(
                os.path.join(model_cache, "scheduler"),
                local_files_only=True
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to load models: {str(e)}")
        
        unet.enable_gradient_checkpointing()
        
        # Create dataset and dataloader
        if not os.path.exists(train_data_dir):
            raise RuntimeError(f"Training data directory not found: {train_data_dir}")
            
        dataset = create_dataset(train_data_dir, "data/coco/annotations/captions_train2014.json")
        transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        
        def transform_examples(examples):
            images = [Image.open(image_path).convert("RGB") for image_path in examples["image"]]
            pixel_values = [
                (torch.cat([transform(image), torch.zeros(1, 256, 256)], dim=0) * 2.0 - 1.0)
                for image in images
            ]
            text_inputs = tokenizer(
                examples["caption"],
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            return {
                "pixel_values": pixel_values,
                "input_ids": text_inputs.input_ids,
                "attention_mask": text_inputs.attention_mask
            }

        train_dataloader = DataLoader(
            dataset.with_transform(transform_examples), 
            batch_size=1, 
            shuffle=True
        )

        # Training setup
        optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-5)
        lr_scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=max_train_steps // 10,
            num_training_steps=max_train_steps
        )
        scaler = torch.amp.GradScaler()

        # Training loop
        unet.train()
        text_encoder.eval()
        progress_bar = tqdm(range(max_train_steps))
        global_step = 0
        final_loss = None

        try:
            while global_step < max_train_steps:
                for batch in train_dataloader:
                    torch.cuda.empty_cache()
                    
                    # Get text embeddings
                    with torch.no_grad():
                        text_embeddings = text_encoder(
                            batch["input_ids"].to("cuda"),
                            attention_mask=batch["attention_mask"].to("cuda")
                        )[0]
                    
                    # Prepare inputs
                    pixel_values = batch["pixel_values"].to("cuda")
                    noise = torch.randn_like(pixel_values)
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (pixel_values.shape[0],)).long().to("cuda")
                    noisy_pixel_values = noise_scheduler.add_noise(pixel_values, noise, timesteps)
                    
                    del batch, pixel_values
                    torch.cuda.empty_cache()

                    # Training step
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        noise_pred = unet(noisy_pixel_values, timesteps, encoder_hidden_states=text_embeddings).sample
                        loss = torch.nn.functional.mse_loss(noise_pred, noise)
                        final_loss = loss.item()

                    del noisy_pixel_values, noise_pred
                    torch.cuda.empty_cache()

                    # Optimization step
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    # Save checkpoint
                    if (global_step + 1) % save_steps == 0:
                        checkpoint_dir = f"{output_dir}/checkpoint"
                        if os.path.exists(checkpoint_dir):
                            shutil.rmtree(checkpoint_dir)
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        unet.save_pretrained(f"{checkpoint_dir}/unet")
                        print(f"Step {global_step+1}/{max_train_steps}: Loss: {loss.item():.4f}")
                        
                        del loss
                        torch.cuda.empty_cache()

                    progress_bar.update(1)
                    global_step += 1
                    if global_step >= max_train_steps:
                        break

        except Exception as e:
            print(f"Error during training: {str(e)}")
            # Save model even if training fails
            unet.save_pretrained(f"{output_dir}/unet")
            final_loss = float('nan') if final_loss is None else final_loss
        finally:
            # Always save the final model
            unet.save_pretrained(f"{output_dir}/unet")
            torch.cuda.empty_cache()
        
        print(f"Finished training for CFG {cfg_scale}, Gen {gen_number}")
        print(f"Model saved to: {output_dir}/unet")
        return final_loss
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return float('nan')

def generate_images(model_path, cfg_scale, gen_number, num_images=100, steps=50):
    """Generate images using the finetuned model for specific generation"""
    try:
        print(f"\nGenerating images for Generation {gen_number} using model finetuned on CFG {cfg_scale}, {num_images} images")
        
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        
        output_dir = f"data/coco/sd_to_sd_cfg_{int(cfg_scale)}_steps_{steps}_gen_{gen_number}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Load COCO captions and prepare image generation list
        coco_captions_file = "data/coco/annotations/captions_train2014.json"
        if not os.path.exists(coco_captions_file):
            raise RuntimeError(f"COCO captions file not found: {coco_captions_file}")
            
        with open(coco_captions_file, 'r') as f:
            annotations = json.load(f)
        
        # Create a dictionary to store the first caption for each image ID
        image_id_to_first_caption = {}
        count = 0
        for ann in annotations['annotations']:
            image_id = ann['image_id']
            if image_id not in image_id_to_first_caption and count < num_images:  # Only take up to num_images
                image_id_to_first_caption[image_id] = ann['caption']
                count += 1
            if count >= num_images:
                break
        
        # Convert to list of tuples and sort by image_id
        image_id_caption_pairs = sorted(list(image_id_to_first_caption.items()))
        
        # Take only the first num_images items
        selected_pairs = image_id_caption_pairs[:num_images]
        
        # Extract image IDs and captions
        image_ids = [pair[0] for pair in selected_pairs]
        captions = [pair[1] for pair in selected_pairs]
        
        # Check existing valid images
        existing_images = {
            int(img_file.split('_')[-1].split('.')[0]): os.path.join(output_dir, img_file)
            for img_file in os.listdir(output_dir)
            if img_file.endswith(('.jpg', '.png')) 
            and os.path.getsize(os.path.join(output_dir, img_file)) > 1024
        }
        
        # Determine which images need to be generated
        images_to_generate = [
            (i, image_id, captions[i])
            for i, image_id in enumerate(image_ids)
            if image_id not in existing_images
        ]
        
        if not images_to_generate:
            print(f"All {num_images} images already exist in {output_dir}")
            return
        
        print(f"Need to generate {len(images_to_generate)} images out of {num_images}")
        
        # Load model with offline fallback
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_cache = os.path.join(cache_dir, "models--CompVis--stable-diffusion-v1-4", "snapshots", "39593d5650112b4cc580433f6b0435385882d819")
        
        if not os.path.exists(model_cache):
            raise RuntimeError(f"Model cache not found at {model_cache}. Please run the script once with internet connection to download the model.")
            
        print(f"Loading models from cache: {model_cache}")
        try:
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_cache,
                torch_dtype=torch.float16,
                local_files_only=True
            ).to("cuda")
            
            if not os.path.exists(model_path):
                raise RuntimeError(f"Model path not found: {model_path}")
                
            pipeline.unet = UNet2DConditionModel.from_pretrained(
                f"{model_path}/unet",
                torch_dtype=torch.float16
            ).to("cuda")
            
            pipeline.set_progress_bar_config(disable=None)
            pipeline.safety_checker = None
        except Exception as e:
            raise RuntimeError(f"Failed to load models: {str(e)}")
        
        # Generate images in batches
        batch_size = 5
        for batch_idx in range(0, len(images_to_generate), batch_size):
            batch_end = min(batch_idx + batch_size, len(images_to_generate))
            print(f"Processing batch {batch_idx//batch_size + 1}/{(len(images_to_generate)-1)//batch_size + 1}")
            
            for i in range(batch_idx, batch_end):
                idx, image_id, caption = images_to_generate[i]
                image_filename = f"COCO_train2014_{image_id:012d}.jpg"
                image_path = os.path.join(output_dir, image_filename)
                
                if os.path.exists(image_path) and os.path.getsize(image_path) > 1024:
                    print(f"  Image {idx+1}/{num_images} (ID: {image_id}) already exists, skipping")
                    continue
                
                print(f"  Generating image {idx+1}/{num_images} (ID: {image_id})")
                
                try:
                    seed = image_id % 10000
                    generator = torch.Generator("cuda").manual_seed(seed)
                    
                    image = pipeline(
                        caption,
                        num_inference_steps=steps,
                        guidance_scale=cfg_scale,
                        generator=generator
                    ).images[0]
                    
                    image.save(image_path)
                    
                    if not os.path.exists(image_path) or os.path.getsize(image_path) < 1024:
                        print(f"  Warning: Failed to save image {image_filename} properly")
                except Exception as e:
                    print(f"  Error generating image {image_id}: {str(e)}")
            
            torch.cuda.empty_cache()
        
        del pipeline
        torch.cuda.empty_cache()
        
        # Verify final results
        valid_images = sum(
            1 for img_file in os.listdir(output_dir)
            if img_file.endswith(('.jpg', '.png')) and os.path.getsize(os.path.join(output_dir, img_file)) > 1024
        )
        
        print(f"Successfully generated {valid_images}/{num_images} valid images in {output_dir}")
        
        if valid_images < num_images:
            print(f"Warning: Only generated {valid_images}/{num_images} valid images")
    
    except Exception as e:
        print(f"Error in image generation: {str(e)}")

def evaluate_model(cfg_scale, gen_number, steps=50):
    """Evaluate the generated images using FID, IS, and CLIP scores"""
    print(f"\nEvaluating generated images for CFG scale {cfg_scale}, Generation {gen_number}")
    
    # Setup paths with steps parameter
    eval_folder = "data/coco/eval"  # Original evaluation images
    gen_folder = f"data/coco/sd_to_sd_cfg_{int(cfg_scale)}_steps_{steps}_gen_{gen_number}"
    annotation_file = "data/coco/annotations/captions_val2014.json"
    
    # Load CLIP model with error handling
    device = "cuda"
    try:
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, download_root=os.path.expanduser("~/.cache/clip"))
    except Exception as e:
        print(f"Warning: Failed to load CLIP model: {str(e)}")
        clip_model = None
    
    # Initialize metrics
    fid = FrechetInceptionDistance(normalize=True).to(device)
    inception_score = InceptionScore(normalize=True).to(device)
    
    # Load and preprocess real and generated images
    def load_images(folder_path):
        images = []
        if not os.path.exists(folder_path):
            print(f"Warning: Image folder not found: {folder_path}")
            return images
            
        for img_file in sorted(os.listdir(folder_path)):
            if img_file.endswith(('.jpg', '.png')):
                img_path = os.path.join(folder_path, img_file)
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = ToTensor()(img)  # Convert to tensor [0,1]
                    img = (img * 255).to(torch.uint8)  # Convert to [0,255] for FID
                    images.append(img)
                except Exception as e:
                    print(f"Warning: Failed to load image {img_path}: {str(e)}")
        return images

    # Load images
    real_images = load_images(eval_folder)
    generated_images = load_images(gen_folder)
    
    if not real_images or not generated_images:
        print("Warning: No images found for evaluation")
        return {"fid": float('nan'), "is_mean": float('nan'), "is_std": float('nan'), "clip_score": float('nan')}
    
    results = {}
    
    # Calculate FID
    try:
        print("Calculating FID score...")
        fid.update(torch.stack(real_images).to(device), real=True)
        fid.update(torch.stack(generated_images).to(device), real=False)
        results["fid"] = float(fid.compute())
    except Exception as e:
        print(f"Warning: Failed to calculate FID score: {str(e)}")
        results["fid"] = float('nan')
    
    # Calculate Inception Score
    try:
        print("Calculating Inception Score...")
        inception_score.update(torch.stack(generated_images).to(device))
        is_mean, is_std = inception_score.compute()
        results["is_mean"] = float(is_mean)
        results["is_std"] = float(is_std)
    except Exception as e:
        print(f"Warning: Failed to calculate Inception Score: {str(e)}")
        results["is_mean"] = float('nan')
        results["is_std"] = float('nan')
    
    # Calculate CLIP score if model is available
    results["clip_score"] = float('nan')
    if clip_model is not None:
        try:
            print("Calculating CLIP score...")
            # Load captions
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
            
            caption_dict = {}
            for ann in annotations['annotations']:
                image_id = ann['image_id']
                caption = ann['caption']
                caption_dict[image_id] = caption
            
            # Calculate CLIP scores
            clip_scores = []
            for img_file in sorted(os.listdir(gen_folder)):
                if img_file.endswith(('.jpg', '.png')):
                    image_id = int(img_file.split('_')[-1].split('.')[0])
                    if image_id in caption_dict:
                        img_path = os.path.join(gen_folder, img_file)
                        try:
                            image = clip_preprocess(Image.open(img_path)).unsqueeze(0).to(device)
                            text = clip.tokenize([caption_dict[image_id]]).to(device)
                            
                            with torch.no_grad():
                                image_features = clip_model.encode_image(image)
                                text_features = clip_model.encode_text(text)
                                
                            similarity = torch.nn.functional.cosine_similarity(image_features, text_features)
                            clip_scores.append(similarity.item())
                        except Exception as e:
                            print(f"Warning: Failed to calculate CLIP score for image {img_path}: {str(e)}")
            
            if clip_scores:
                results["clip_score"] = float(sum(clip_scores) / len(clip_scores))
        except Exception as e:
            print(f"Warning: Failed to calculate CLIP scores: {str(e)}")
    
    return results

def find_last_generation(cfg_scale, steps=50):
    """Find the last completed generation by checking both model and image directories"""
    gen = 1
    while True:
        model_dir = f"models/sd_to_sd_cfg_{int(cfg_scale)}_steps_{steps}_gen_{gen}"
        data_dir = f"data/coco/sd_to_sd_cfg_{int(cfg_scale)}_steps_{steps}_gen_{gen}"
        
        if not os.path.exists(f"{model_dir}/unet") or not os.path.exists(data_dir):
            return gen - 1
        gen += 1

def run_generation_loop(cfg_scale, target_generation=3, steps=50, epochs=5, num_images=100):
    """Run generations until target_generation is reached"""
    results = {}
    eval_results = {}
    
    # Check and create gen_0 (always use 100 images for gen_0)
    if not check_and_create_gen_zero(cfg_scale, num_images=100, steps=steps):
        print("Failed to create Generation 0 images. Cannot proceed.")
        return
    
    # Verify gen_0
    gen_zero_dir = f"data/coco/sd_to_sd_cfg_{int(cfg_scale)}_steps_{steps}_gen_0"
    if len([f for f in os.listdir(gen_zero_dir) if f.endswith(('.jpg', '.png'))]) < 100:
        print("Generation 0 incomplete. Please check and try again.")
        return
    
    # Find last completed generation and continue
    last_gen = find_last_generation(cfg_scale, steps)
    start_gen = last_gen + 1
    
    if start_gen > target_generation:
        print(f"Target generation {target_generation} already completed!")
        return
    
    try:
        for gen in range(start_gen, target_generation + 1):
            print(f"\n{'='*50}\nStarting Generation {gen}\n{'='*50}")
            
            current_model_dir = f"models/sd_to_sd_cfg_{int(cfg_scale)}_steps_{steps}_gen_{gen}"
            previous_model_dir = f"models/sd_to_sd_cfg_{int(cfg_scale)}_steps_{steps}_gen_{gen-1}" if gen > 1 else None
            
            # Train and generate
            results[f"gen_{gen}"] = train_stable_diffusion(cfg_scale, gen, previous_model_dir, steps, epochs)
            generate_images(current_model_dir, cfg_scale, gen, num_images=num_images, steps=steps)
            
            # Evaluate
            try:
                eval_results[f"gen_{gen}"] = evaluate_model(cfg_scale, gen, steps)
            except Exception as e:
                print(f"Evaluation error for Gen {gen}: {str(e)}")
                eval_results[f"gen_{gen}"] = {"fid": float('nan'), "is_mean": float('nan'), 
                                            "is_std": float('nan'), "clip_score": float('nan')}
            
            torch.cuda.empty_cache()
    
    except Exception as e:
        print(f"Generation loop error: {str(e)}")
    finally:
        # Print results
        print("\nTraining Results:")
        print("="*50)
        for gen, loss in results.items():
            print(f"{gen:^10} | CFG {cfg_scale:^9} | Steps {steps:^5} | Loss {loss:.4f}")
        
        if eval_results:
            print("\nEvaluation Results:")
            print("="*80)
            for gen, metrics in eval_results.items():
                print(f"{gen:^10} | CFG {cfg_scale:^9} | Steps {steps:^5} | "
                      f"FID {metrics['fid']:^9.2f} | IS {metrics['is_mean']:^14.2f} | "
                      f"CLIP {metrics['clip_score']:^10.4f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Stable Diffusion with multiple generations")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="CFG scale for training")
    parser.add_argument("--target_gen", type=int, default=3, help="Target generation to reach")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--num_images", type=int, default=100, help="Number of images to generate for each generation (except gen_0 which always uses 100)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    
    args = parser.parse_args()
    run_generation_loop(args.cfg_scale, args.target_gen, args.steps, args.epochs, args.num_images)

