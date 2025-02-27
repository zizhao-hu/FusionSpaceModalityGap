import os
import argparse
import glob
import json
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import random
import platform
import re

def load_coco_captions(annotation_file="data/coco/annotations/captions_train2014.json"):
    """Load COCO captions from json file"""
    print(f"Loading captions from {annotation_file}")
    try:
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
        
        print(f"Loaded {len(image_id_to_caption)} image captions")
        return image_id_to_caption
    except Exception as e:
        print(f"Error loading COCO captions: {str(e)}")
        return {}

def extract_image_id_from_path(image_path):
    """Extract the COCO image ID from a file path"""
    try:
        # Extract filename from path
        filename = os.path.basename(image_path)
        
        # Extract image ID using regex (looking for pattern like COCO_train2014_000000000001.jpg)
        match = re.search(r'COCO_train2014_(\d+)', filename)
        if match:
            image_id = int(match.group(1))
            return image_id
        return None
    except Exception as e:
        print(f"Error extracting image ID: {str(e)}")
        return None

def find_image_by_index(index, cfg_scale, gen_number):
    """Find an image by its index in a specific generation folder"""
    if index < 0 or index > 99:
        raise ValueError("Index must be between 0 and 99")
    
    # Construct the path to the generation folder
    gen_dir = f"data/coco/sd_to_sd_cfg_{int(cfg_scale)}_gen_{gen_number}"
    
    if not os.path.exists(gen_dir):
        return None
    
    # Get all image files in the directory
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(glob.glob(f"{gen_dir}/*{ext}"))
    
    # Sort the files to ensure consistent ordering
    image_files.sort()
    
    # Check if the index is valid
    if index >= len(image_files):
        return None
    
    return image_files[index]

def find_images_across_generations(index, cfg_scale, max_gen=10):
    """Find the same image (by index) across all generations"""
    matching_images = []
    
    # Check for generation 0 first
    gen_0_image = find_image_by_index(index, cfg_scale, 0)
    if gen_0_image:
        matching_images.append((0, gen_0_image))
    
    # Check for all other generations
    for gen in range(1, max_gen + 1):
        gen_image = find_image_by_index(index, cfg_scale, gen)
        if gen_image:
            matching_images.append((gen, gen_image))
        else:
            # If we can't find an image in this generation, check if the directory exists
            gen_dir = f"data/coco/sd_to_sd_cfg_{int(cfg_scale)}_gen_{gen}"
            if not os.path.exists(gen_dir):
                break  # No more generations found
    
    return matching_images

def generate_images_from_caption(caption, cfg_scale, generations=None, seed=None, output_dir="generated_comparisons", num_rows=1, batch_size=4):
    """Generate images from a caption using models from different generations
    
    Args:
        caption: The text prompt to generate images from
        cfg_scale: The classifier-free guidance scale
        generations: List of specific generations to use (default: all available)
        seed: Random seed for generation (default: random)
        output_dir: Directory to save generated images
        num_rows: Number of rows to generate with different noise seeds (default: 1)
        batch_size: Number of rows to process in a single batch (default: 4)
    """
    if generations is None:
        # Find all available generations
        generations = []
        gen = 0
        
        # Check for base model (generation 0)
        base_model_exists = True
        generations.append(0)
        
        # Check for fine-tuned models
        while True:
            gen += 1
            model_dir = f"models/sd_to_sd_cfg_{int(cfg_scale)}_gen_{gen}/unet"
            if os.path.exists(model_dir):
                generations.append(gen)
                print(f"Found model for generation {gen} at {model_dir}")
            else:
                print(f"No model found for generation {gen} at {model_dir}")
                break
    
    if not generations:
        print("WARNING: No generation models found. Using only the base model (generation 0).")
        generations = [0]
    
    print(f"Will generate images for generations: {generations}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate seeds for all rows
    row_seeds = []
    if seed is not None:
        # Use the provided seed for the first row
        row_seeds.append(seed)
    
    # Generate random seeds for remaining rows
    for i in range(len(row_seeds), num_rows):
        row_seeds.append(random.randint(0, 2**32 - 1))
    
    print(f"Using seeds for {num_rows} rows: {row_seeds}")
    
    # Process rows one at a time to avoid memory issues
    all_generated_rows = []
    
    # Process each generation for all rows
    for gen in generations:
        print(f"\nGenerating images for generation {gen}...")
        
        try:
            # Load the pipeline
            if gen == 0:
                # Use the original model for generation 0
                print("Loading original Stable Diffusion v1.4 model...")
                pipeline = StableDiffusionPipeline.from_pretrained(
                    "CompVis/stable-diffusion-v1-4",
                    torch_dtype=torch.float16
                ).to("cuda")
            else:
                # Load the base model first
                print("Loading base Stable Diffusion v1.4 model...")
                pipeline = StableDiffusionPipeline.from_pretrained(
                    "CompVis/stable-diffusion-v1-4",
                    torch_dtype=torch.float16
                ).to("cuda")
                
                # Then load the fine-tuned UNet
                model_path = f"models/sd_to_sd_cfg_{int(cfg_scale)}_gen_{gen}/unet"
                print(f"Loading UNet from {model_path}")
                
                if not os.path.exists(model_path):
                    print(f"ERROR: UNet model path {model_path} does not exist!")
                    print("Skipping this generation. Please train the model first.")
                    continue
                
                # Load the fine-tuned UNet
                pipeline.unet = UNet2DConditionModel.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16
                ).to("cuda")
            
            # Disable safety checker for consistency
            pipeline.safety_checker = None
            
            # Set number of inference steps
            num_inference_steps = 50
            pipeline.scheduler.set_timesteps(num_inference_steps)
            
            # Process rows in batches
            for batch_start in range(0, num_rows, batch_size):
                batch_end = min(batch_start + batch_size, num_rows)
                current_batch_size = batch_end - batch_start
                batch_seeds = row_seeds[batch_start:batch_end]
                
                print(f"Processing batch of {current_batch_size} rows (rows {batch_start+1}-{batch_end} of {num_rows})")
                print(f"Batch seeds: {batch_seeds}")
                
                # Process each seed in the batch individually to avoid issues
                for i, row_seed in enumerate(batch_seeds):
                    row_idx = batch_start + i
                    print(f"  Generating image for row {row_idx+1} with seed {row_seed}")
                    
                    # Set the generator with the specific seed
                    generator = torch.Generator("cuda").manual_seed(row_seed)
                    
                    try:
                        # Generate the image
                        print(f"    Running pipeline with caption: '{caption}'")
                        result = pipeline(
                            caption,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=cfg_scale,
                            generator=generator
                        )
                        
                        image = result.images[0]
                        
                        # Save the image
                        output_path = os.path.join(output_dir, f"gen_{gen}_seed_{row_seed}_row_{row_idx}.png")
                        image.save(output_path)
                        print(f"    Saved image to {output_path}")
                        
                        # Add to results for this row
                        row_exists = False
                        for j, (existing_seed, row_images) in enumerate(all_generated_rows):
                            if existing_seed == row_seed:
                                all_generated_rows[j][1].append((gen, output_path))
                                row_exists = True
                                break
                        
                        if not row_exists:
                            all_generated_rows.append((row_seed, [(gen, output_path)]))
                    
                    except Exception as e:
                        print(f"    ERROR generating image for row {row_idx+1}, seed {row_seed}: {str(e)}")
            
            # Clean up to save memory
            del pipeline
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error setting up pipeline for generation {gen}: {str(e)}")
    
    # Sort the rows by seed for consistency
    all_generated_rows.sort(key=lambda x: x[0])
    
    if not all_generated_rows:
        print("WARNING: Failed to generate any images. Please check the error messages above.")
    
    return all_generated_rows

def get_libertine_font(size=80):
    """Try to load Libertine font from common locations based on the operating system with a much larger default size"""
    # Print the font size being requested for debugging
    print(f"Attempting to load font with size: {size}")
    
    font_paths = []
    
    # Add common Libertine font locations based on OS
    system = platform.system()
    if system == "Windows":
        font_paths = [
            "C:/Windows/Fonts/LinLibertine_R.ttf",
            "C:/Windows/Fonts/Linux Libertine.ttf",
            "C:/Windows/Fonts/LinuxLibertine-Regular.ttf",
            "C:/Windows/Fonts/times.ttf",  # Times New Roman
            "C:/Windows/Fonts/timesbd.ttf",  # Times New Roman Bold
            "C:/Windows/Fonts/arial.ttf",  # Arial as fallback
            "C:/Windows/Fonts/georgia.ttf"  # Georgia as fallback
        ]
    elif system == "Darwin":  # macOS
        font_paths = [
            "/Library/Fonts/LinLibertine_R.ttf",
            "/Library/Fonts/Linux Libertine.ttf",
            "/System/Library/Fonts/LinLibertine_R.ttf",
            "/Library/Fonts/Times New Roman.ttf",
            "/System/Library/Fonts/Times.ttc",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Georgia.ttf"
        ]
    else:  # Linux and others
        font_paths = [
            "/usr/share/fonts/opentype/linux-libertine/LinLibertine_R.otf",
            "/usr/share/fonts/truetype/linux-libertine/LinLibertine_R.ttf",
            "/usr/share/fonts/TTF/LinLibertine_R.ttf",
            "/usr/share/fonts/TTF/times.ttf",
            "/usr/share/fonts/TTF/DejaVuSerif.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf"
        ]
    
    # Try to load the font from the paths
    for path in font_paths:
        try:
            font = ImageFont.truetype(path, size)
            print(f"Successfully loaded font from {path} with size {size}")
            return font
        except Exception as e:
            continue
    
    # If we couldn't load any font, create a default font with the specified size
    print(f"Could not load any font from paths. Using default font.")
    try:
        # Try to load a default font with the specified size
        default_font = ImageFont.load_default()
        
        # For PIL versions that support it, try to get a larger default font
        if hasattr(default_font, "getsize"):
            # This is a crude way to increase the default font size
            # by creating a larger image and drawing text on it
            img = Image.new('RGB', (500, 500), color='white')
            draw = ImageDraw.Draw(img)
            
            # Draw a large character and measure its size
            draw.text((10, 10), "X", fill="black", font=default_font)
            
            # Return the default font, which is the best we can do
            print("Using default font with crude size adjustment")
            return default_font
        else:
            print("Using basic default font")
            return default_font
    except Exception as e:
        print(f"Error creating default font: {str(e)}")
        # Last resort - return None and let the calling code handle it
        return None

def create_comparison_image(matching_images, output_path=None, cfg_scale=None, index=None, caption=None):
    """Create a comparison image with all generations side by side with spacing"""
    if not matching_images:
        print("No matching images found across generations.")
        return
    
    # If no caption is provided but we have image paths, try to extract the caption from COCO dataset
    if caption is None and matching_images:
        # Get the first image path to extract the image ID
        _, first_image_path = matching_images[0]
        image_id = extract_image_id_from_path(first_image_path)
        
        if image_id:
            # Load COCO captions
            coco_captions = load_coco_captions()
            
            # Get caption for this image ID
            if image_id in coco_captions and coco_captions[image_id]:
                caption = coco_captions[image_id][0]  # Use the first caption
                print(f"Found caption for image ID {image_id}: {caption}")
    
    # Open all images
    images = []
    for gen, path in matching_images:
        try:
            img = Image.open(path)
            images.append((gen, img))
        except Exception as e:
            print(f"Error opening image for generation {gen}: {str(e)}")
    
    if not images:
        print("Failed to open any images.")
        return
    
    # Get dimensions of the first image
    _, first_img = images[0]
    img_width, img_height = first_img.size
    
    # Add spacing between images
    image_spacing = 40  # pixels between images
    
    # Calculate total width with spacing
    total_width = (img_width * len(images)) + (image_spacing * (len(images) - 1))
    
    # Add extra height for caption and generation labels
    gen_label_height = 100  # Space for generation labels above images
    caption_height = 200 if caption else 0  # Space for caption
    
    # Create a new image with all generations side by side
    comparison_img = Image.new('RGB', 
                              (total_width, img_height + gen_label_height + caption_height), 
                              color='white')
    
    # Get fonts - DRASTICALLY INCREASED SIZES
    # Bold font for generation labels
    try:
        # Try to load a bold font for generation labels
        system = platform.system()
        if system == "Windows":
            label_font = ImageFont.truetype("C:/Windows/Fonts/timesbd.ttf", 70)  # Times New Roman Bold
        elif system == "Darwin":  # macOS
            label_font = ImageFont.truetype("/Library/Fonts/Times New Roman Bold.ttf", 70)
        else:  # Linux
            label_font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSerif-Bold.ttf", 70)
        print("Successfully loaded bold font for generation labels")
    except Exception as e:
        # Fall back to regular font if bold is not available
        label_font = get_libertine_font(70)
        print(f"Could not load bold font, using regular font: {str(e)}")
    
    # Italic font for caption
    try:
        # Try to load an italic font for captions
        system = platform.system()
        if system == "Windows":
            caption_font = ImageFont.truetype("C:/Windows/Fonts/timesi.ttf", 80)  # Times New Roman Italic
        elif system == "Darwin":  # macOS
            caption_font = ImageFont.truetype("/Library/Fonts/Times New Roman Italic.ttf", 80)
        else:  # Linux
            caption_font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSerif-Italic.ttf", 80)
        print("Successfully loaded italic font for captions")
    except Exception as e:
        # Fall back to regular font if italic is not available
        caption_font = get_libertine_font(80)
        print(f"Could not load italic font, using regular font: {str(e)}")
    
    draw = ImageDraw.Draw(comparison_img)
    
    # Add caption if provided (at the bottom)
    if caption:
        # Draw a light gray background for the caption
        draw.rectangle([(0, gen_label_height + img_height), (total_width, gen_label_height + img_height + caption_height)], fill=(245, 245, 245))
        
        # Draw caption with word wrapping (in italic)
        caption_text = f"\"{caption}\""  # Just the caption in quotes, no index
        caption_position = (40, gen_label_height + img_height + 30)  # Position below images
        
        # Simple word wrapping
        words = caption_text.split()
        lines = []
        current_line = []
        current_width = 0
        max_width = total_width - 80  # 40px padding on each side
        
        for word in words:
            try:
                word_width = draw.textlength(word + " ", font=caption_font) if hasattr(draw, 'textlength') else caption_font.getsize(word + " ")[0]
            except:
                # Fallback if font measurement fails
                word_width = len(word) * 25  # Rough estimate
                
            if current_width + word_width <= max_width:
                current_line.append(word)
                current_width += word_width
            else:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_width = word_width
        
        if current_line:
            lines.append(" ".join(current_line))
        
        # Draw each line
        for i, line in enumerate(lines):
            line_y = caption_position[1] + (i * 80)  # Line spacing
            draw.text((caption_position[0], line_y), line, fill="black", font=caption_font)
    
    # Paste all images side by side with spacing and add generation labels above
    for i, (gen, img) in enumerate(images):
        # Calculate positions
        x_position = i * (img_width + image_spacing)
        y_position = gen_label_height  # Images start after generation labels
        
        # Add generation number as bold text ABOVE the image
        text = f"Generation {gen}"
        try:
            text_width = draw.textlength(text, font=label_font) if hasattr(draw, 'textlength') else label_font.getsize(text)[0]
        except:
            # Fallback if font measurement fails
            text_width = len(text) * 20  # Rough estimate
            
        text_position = (x_position + (img_width - text_width) // 2, 30)  # Centered above image
        draw.text(text_position, text, fill="black", font=label_font)
        
        # Paste the image
        comparison_img.paste(img, (x_position, y_position))
    
    # Save the comparison image
    if output_path is None:
        if index is not None:
            output_path = f"comparison_cfg_{cfg_scale}_index_{index}"
        elif caption is not None:
            # Create a safe filename from the caption
            safe_caption = "".join(c if c.isalnum() else "_" for c in caption[:30])
            output_path = f"comparison_cfg_{cfg_scale}_caption_{safe_caption}"
        else:
            output_path = f"comparison_cfg_{cfg_scale}"
    
    # Ensure the output path has an extension
    if not output_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        output_path += '.png'
    
    comparison_img.save(output_path)
    print(f"Comparison image saved to: {output_path}")
    
    # Display the image
    plt.figure(figsize=(15, 5))
    plt.imshow(np.array(comparison_img))
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return output_path

def create_grid_comparison(indices, cfg_scale, output_path=None, max_gen=10):
    """Create a grid of images comparing multiple indices across generations"""
    # Load COCO captions
    coco_captions = load_coco_captions()
    
    # Find matching images for each index
    all_matching_sets = []
    for index in indices:
        matching_images = find_images_across_generations(index, cfg_scale, max_gen)
        if matching_images:
            # Try to get the caption for this image
            caption = None
            _, first_image_path = matching_images[0]
            image_id = extract_image_id_from_path(first_image_path)
            if image_id and image_id in coco_captions and coco_captions[image_id]:
                caption = coco_captions[image_id][0]  # Use the first caption
            
            all_matching_sets.append((index, matching_images, caption))
    
    if not all_matching_sets:
        print("No matching images found across generations for any selected index.")
        return
    
    # Open all images
    image_grid = []
    for index, matching_set, caption in all_matching_sets:
        row_images = []
        # Create a dictionary mapping generation number to image path
        gen_to_path = {gen: path for gen, path in matching_set}
        
        # Find all unique generation numbers across all sets
        all_gens = sorted(set(gen for gen, _ in matching_set))
        
        for gen in all_gens:
            if gen in gen_to_path:
                try:
                    img = Image.open(gen_to_path[gen])
                    row_images.append((gen, img))
                except Exception as e:
                    print(f"Error opening image for index {index}, generation {gen}: {str(e)}")
        
        if row_images:
            image_grid.append((index, row_images, caption))
    
    if not image_grid:
        print("Failed to open any images.")
        return
    
    # Get dimensions of the first image
    _, first_row, _ = image_grid[0]
    _, first_img = first_row[0]
    img_width, img_height = first_img.size
    
    # Find the maximum number of generations in any row
    max_cols = max(len(row) for _, row, _ in image_grid)
    
    # Add spacing between images
    image_spacing = 40  # pixels between images
    row_spacing = 80    # extra pixels between rows
    
    # Calculate total width with spacing
    total_width = (img_width * max_cols) + (image_spacing * (max_cols - 1))
    
    # Calculate height for each row (image + caption + spacing)
    caption_height = 150  # Space for caption
    row_height = img_height + caption_height + row_spacing
    
    # Create a new image with all generations in a grid
    gen_label_height = 100  # Space for generation labels at the top
    total_height = (row_height * len(image_grid)) + gen_label_height  # Extra space for column headers
    grid_img = Image.new('RGB', (total_width, total_height), color='white')
    
    # Get fonts - DRASTICALLY INCREASED SIZES
    # Bold font for generation labels
    try:
        # Try to load a bold font for generation labels
        system = platform.system()
        if system == "Windows":
            label_font = ImageFont.truetype("C:/Windows/Fonts/timesbd.ttf", 70)  # Times New Roman Bold
        elif system == "Darwin":  # macOS
            label_font = ImageFont.truetype("/Library/Fonts/Times New Roman Bold.ttf", 70)
        else:  # Linux
            label_font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSerif-Bold.ttf", 70)
        print("Successfully loaded bold font for generation labels")
    except Exception as e:
        # Fall back to regular font if bold is not available
        label_font = get_libertine_font(70)
        print(f"Could not load bold font, using regular font: {str(e)}")
    
    # Italic font for caption
    try:
        # Try to load an italic font for captions
        system = platform.system()
        if system == "Windows":
            caption_font = ImageFont.truetype("C:/Windows/Fonts/timesi.ttf", 80)  # Times New Roman Italic
        elif system == "Darwin":  # macOS
            caption_font = ImageFont.truetype("/Library/Fonts/Times New Roman Italic.ttf", 80)
        else:  # Linux
            caption_font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSerif-Italic.ttf", 80)
        print("Successfully loaded italic font for captions")
    except Exception as e:
        # Fall back to regular font if italic is not available
        caption_font = get_libertine_font(80)
        print(f"Could not load italic font, using regular font: {str(e)}")
    
    draw = ImageDraw.Draw(grid_img)
    
    # Draw generation labels at the top
    for col_idx, gen in enumerate(sorted(set(gen for _, row, _ in image_grid for gen, _ in row))):
        x_position = col_idx * (img_width + image_spacing)
        gen_text = f"Generation {gen}"
        try:
            gen_text_width = draw.textlength(gen_text, font=label_font) if hasattr(draw, 'textlength') else label_font.getsize(gen_text)[0]
        except:
            # Fallback if font measurement fails
            gen_text_width = len(gen_text) * 20  # Rough estimate
            
        gen_text_position = (x_position + (img_width - gen_text_width) // 2, 30)  # Centered above column
        draw.text(gen_text_position, gen_text, fill="black", font=label_font)
    
    # Paste all images in a grid
    for row_idx, (index, row_images, caption) in enumerate(image_grid):
        # Calculate row position
        row_y = gen_label_height + (row_idx * row_height)  # Start after generation labels
        
        # Draw caption for this row
        if caption:
            # Draw a light gray background for the caption
            caption_bg_height = caption_height - 10
            draw.rectangle([(0, row_y + img_height), (total_width, row_y + img_height + caption_bg_height)], fill=(245, 245, 245))
            
            # Draw caption with word wrapping (in italic, without index)
            caption_text = f"\"{caption}\""  # Just the caption in quotes, no index
            caption_position = (40, row_y + img_height + 20)  # Position below images
            
            # Simple word wrapping
            words = caption_text.split()
            lines = []
            current_line = []
            current_width = 0
            max_width = total_width - 80  # 40px padding on each side
            
            for word in words:
                try:
                    word_width = draw.textlength(word + " ", font=caption_font) if hasattr(draw, 'textlength') else caption_font.getsize(word + " ")[0]
                except:
                    # Fallback if font measurement fails
                    word_width = len(word) * 25  # Rough estimate
                    
                if current_width + word_width <= max_width:
                    current_line.append(word)
                    current_width += word_width
                else:
                    lines.append(" ".join(current_line))
                    current_line = [word]
                    current_width = word_width
            
            if current_line:
                lines.append(" ".join(current_line))
            
            # Draw each line
            for i, line in enumerate(lines):
                line_y = caption_position[1] + (i * 80)  # Line spacing
                draw.text((caption_position[0], line_y), line, fill="black", font=caption_font)
        else:
            # If no caption, leave the space empty
            pass
        
        # Paste images for this row
        for col_idx, (gen, img) in enumerate(row_images):
            # Calculate image position
            x_position = col_idx * (img_width + image_spacing)
            y_position = row_y  # Images at the top of each row
            
            # Paste the image
            grid_img.paste(img, (x_position, y_position))
    
    # Save the grid image
    if output_path is None:
        indices_str = "_".join(str(idx) for idx in indices)
        output_path = f"grid_comparison_cfg_{cfg_scale}_indices_{indices_str}"
    
    # Ensure the output path has an extension
    if not output_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        output_path += '.png'
    
    grid_img.save(output_path)
    print(f"Grid comparison image saved to: {output_path}")
    
    # Display the image
    plt.figure(figsize=(15, 10))
    plt.imshow(np.array(grid_img))
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return output_path

def create_multi_row_comparison(all_rows, output_path=None, cfg_scale=None, caption=None):
    """Create a comparison image with multiple rows, each row representing a different seed"""
    if not all_rows:
        print("No images generated.")
        return
    
    # Open all images
    processed_rows = []
    for row_seed, row_images in all_rows:
        images = []
        for gen, path in row_images:
            try:
                img = Image.open(path)
                images.append((gen, img))
            except Exception as e:
                print(f"Error opening image for seed {row_seed}, generation {gen}: {str(e)}")
        
        if images:
            processed_rows.append((row_seed, images))
    
    if not processed_rows:
        print("Failed to open any images.")
        return
    
    # Get dimensions of the first image
    _, first_row = processed_rows[0]
    _, first_img = first_row[0]
    img_width, img_height = first_img.size
    
    # Add spacing between images
    image_spacing = 40  # pixels between images
    row_spacing = 80    # pixels between rows
    
    # Calculate total width with spacing
    max_cols = max(len(row) for _, row in processed_rows)
    total_width = (img_width * max_cols) + (image_spacing * (max_cols - 1))
    
    # Add extra height for caption and generation labels
    gen_label_height = 100  # Space for generation labels above images
    caption_height = 200 if caption else 0  # Space for caption at the bottom
    seed_label_height = 80  # Space for seed labels at the start of each row
    
    # Calculate total height
    total_height = (img_height * len(processed_rows)) + (row_spacing * (len(processed_rows) - 1)) + gen_label_height + caption_height
    
    # Create a new image with all generations and seeds
    comparison_img = Image.new('RGB', (total_width, total_height), color='white')
    
    # Get fonts
    # Bold font for generation labels and seed labels
    try:
        # Try to load a bold font for labels
        system = platform.system()
        if system == "Windows":
            label_font = ImageFont.truetype("C:/Windows/Fonts/timesbd.ttf", 70)  # Times New Roman Bold
        elif system == "Darwin":  # macOS
            label_font = ImageFont.truetype("/Library/Fonts/Times New Roman Bold.ttf", 70)
        else:  # Linux
            label_font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSerif-Bold.ttf", 70)
        print("Successfully loaded bold font for labels")
    except Exception as e:
        # Fall back to regular font if bold is not available
        label_font = get_libertine_font(70)
        print(f"Could not load bold font, using regular font: {str(e)}")
    
    # Italic font for caption
    try:
        # Try to load an italic font for captions
        system = platform.system()
        if system == "Windows":
            caption_font = ImageFont.truetype("C:/Windows/Fonts/timesi.ttf", 80)  # Times New Roman Italic
        elif system == "Darwin":  # macOS
            caption_font = ImageFont.truetype("/Library/Fonts/Times New Roman Italic.ttf", 80)
        else:  # Linux
            caption_font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSerif-Italic.ttf", 80)
        print("Successfully loaded italic font for captions")
    except Exception as e:
        # Fall back to regular font if italic is not available
        caption_font = get_libertine_font(80)
        print(f"Could not load italic font, using regular font: {str(e)}")
    
    draw = ImageDraw.Draw(comparison_img)
    
    # Draw generation labels at the top
    all_gens = sorted(set(gen for _, row in processed_rows for gen, _ in row))
    for col_idx, gen in enumerate(all_gens):
        x_position = col_idx * (img_width + image_spacing)
        gen_text = f"Generation {gen}"
        try:
            gen_text_width = draw.textlength(gen_text, font=label_font) if hasattr(draw, 'textlength') else label_font.getsize(gen_text)[0]
        except:
            # Fallback if font measurement fails
            gen_text_width = len(gen_text) * 20  # Rough estimate
            
        gen_text_position = (x_position + (img_width - gen_text_width) // 2, 30)  # Centered above column
        draw.text(gen_text_position, gen_text, fill="black", font=label_font)
    
    # Paste all images in a grid
    for row_idx, (row_seed, row_images) in enumerate(processed_rows):
        # Calculate row position
        row_y = gen_label_height + (row_idx * (img_height + row_spacing))
        
        # Draw seed label for this row
        seed_text = f"Seed: {row_seed}"
        try:
            seed_text_width = draw.textlength(seed_text, font=label_font) if hasattr(draw, 'textlength') else label_font.getsize(seed_text)[0]
        except:
            # Fallback if font measurement fails
            seed_text_width = len(seed_text) * 20  # Rough estimate
        
        # Draw seed label at the beginning of the row
        draw.text((20, row_y + (img_height // 2) - 30), seed_text, fill="black", font=label_font)
        
        # Paste images for this row
        for col_idx, (gen, img) in enumerate(row_images):
            # Calculate image position
            x_position = col_idx * (img_width + image_spacing)
            y_position = row_y
            
            # Paste the image
            comparison_img.paste(img, (x_position, y_position))
    
    # Add caption if provided (at the bottom)
    if caption:
        # Draw a light gray background for the caption
        caption_y = gen_label_height + (len(processed_rows) * (img_height + row_spacing)) - row_spacing
        draw.rectangle([(0, caption_y), (total_width, caption_y + caption_height)], fill=(245, 245, 245))
        
        # Draw caption with word wrapping (in italic)
        caption_text = f"\"{caption}\""  # Just the caption in quotes
        caption_position = (40, caption_y + 30)
        
        # Simple word wrapping
        words = caption_text.split()
        lines = []
        current_line = []
        current_width = 0
        max_width = total_width - 80  # 40px padding on each side
        
        for word in words:
            try:
                word_width = draw.textlength(word + " ", font=caption_font) if hasattr(draw, 'textlength') else caption_font.getsize(word + " ")[0]
            except:
                # Fallback if font measurement fails
                word_width = len(word) * 25  # Rough estimate
                
            if current_width + word_width <= max_width:
                current_line.append(word)
                current_width += word_width
            else:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_width = word_width
        
        if current_line:
            lines.append(" ".join(current_line))
        
        # Draw each line
        for i, line in enumerate(lines):
            line_y = caption_position[1] + (i * 80)  # Line spacing
            draw.text((caption_position[0], line_y), line, fill="black", font=caption_font)
    
    # Save the comparison image
    if output_path is None:
        if caption is not None:
            # Create a safe filename from the caption
            safe_caption = "".join(c if c.isalnum() else "_" for c in caption[:30])
            output_path = f"multi_row_comparison_cfg_{cfg_scale}_caption_{safe_caption}"
        else:
            output_path = f"multi_row_comparison_cfg_{cfg_scale}"
    
    # Ensure the output path has an extension
    if not output_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        output_path += '.png'
    
    comparison_img.save(output_path)
    print(f"Multi-row comparison image saved to: {output_path}")
    
    # Display the image
    plt.figure(figsize=(15, 10))
    plt.imshow(np.array(comparison_img))
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Compare generations by index or generate new images from a caption")
    parser.add_argument("--cfg_scale", type=float, default=7.0, help="CFG scale to use for finding/generating images")
    parser.add_argument("--index", type=int, help="Index of the image to compare (0-99)")
    parser.add_argument("--indices", type=int, nargs="+", help="Multiple indices to compare in a grid")
    parser.add_argument("--caption", type=str, help="Caption to generate new images with across all generations")
    parser.add_argument("--generations", type=int, nargs="+", help="Specific generations to use (default: all available)")
    parser.add_argument("--seed", type=int, help="Random seed for generation (default: random)")
    parser.add_argument("--output", type=str, help="Output path for the comparison image (optional)")
    parser.add_argument("--max_gen", type=int, default=10, help="Maximum generation number to check")
    parser.add_argument("--output_dir", type=str, default="generated_comparisons", help="Directory to save generated images")
    parser.add_argument("--no_caption", action="store_true", help="Disable automatic caption extraction for existing images")
    parser.add_argument("--num_rows", type=int, default=1, help="Number of rows to generate with different noise seeds (for caption mode)")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of rows to process in a single batch (default: 4)")
    
    args = parser.parse_args()
    
    # Handle caption generation mode
    if args.caption:
        print(f"Generating images from caption: '{args.caption}' with CFG scale {args.cfg_scale}")
        print(f"Creating {args.num_rows} row(s) with different noise seeds (batch size: {args.batch_size})")
        
        all_rows = generate_images_from_caption(
            args.caption, 
            args.cfg_scale, 
            args.generations, 
            args.seed,
            args.output_dir,
            args.num_rows,
            args.batch_size
        )
        
        if not all_rows:
            print("Failed to generate any images.")
            return
        
        total_images = sum(len(row_images) for _, row_images in all_rows)
        print(f"Generated {total_images} images across {len(all_rows)} rows")
        
        if args.num_rows > 1:
            # Create multi-row comparison for multiple seeds
            create_multi_row_comparison(all_rows, args.output, args.cfg_scale, args.caption)
        else:
            # Use the original single-row comparison for a single seed
            create_comparison_image(all_rows[0][1], args.output, args.cfg_scale, caption=args.caption)
        
        return
    
    # Handle multiple indices (grid mode)
    if args.indices:
        # Validate indices
        for idx in args.indices:
            if idx < 0 or idx > 99:
                print(f"Error: Index {idx} is out of range (must be 0-99)")
                return
        
        print(f"Creating grid comparison for indices {args.indices} with CFG scale {args.cfg_scale}...")
        print("Original captions will be automatically extracted and displayed (use --no_caption to disable)")
        create_grid_comparison(args.indices, args.cfg_scale, args.output, args.max_gen)
        return
    
    # Handle single index
    if args.index is None:
        print("Error: Please specify an index (--index), multiple indices (--indices), or a caption (--caption)")
        return
    
    # Validate index
    if args.index < 0 or args.index > 99:
        print(f"Error: Index {args.index} is out of range (must be 0-99)")
        return
    
    print(f"Finding image with index {args.index} across generations with CFG scale {args.cfg_scale}...")
    
    # Find matching images across generations
    matching_images = find_images_across_generations(args.index, args.cfg_scale, args.max_gen)
    
    if not matching_images:
        print(f"No images found with index {args.index} across generations for CFG scale {args.cfg_scale}.")
        return
    
    print(f"Found {len(matching_images)} matching images across generations:")
    for gen, path in matching_images:
        print(f"  Generation {gen}: {path}")
    
    # Create and save the comparison image
    if args.no_caption:
        print("Caption extraction disabled")
        create_comparison_image(matching_images, args.output, args.cfg_scale, args.index, caption=None)
    else:
        print("Original caption will be automatically extracted and displayed")
        create_comparison_image(matching_images, args.output, args.cfg_scale, args.index)

if __name__ == "__main__":
    main() 