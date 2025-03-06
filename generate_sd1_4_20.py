import os 
import json
import torch
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

# -------------------------------
# 1. Set up device and load Stable Diffusion pipeline
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16,
)
pipe = pipe.to(device)
# Disable safety checker
pipe.safety_checker = lambda images, clip_input: (images, [False] * len(images))

# -------------------------------
# 2. Define paths and parameters
# -------------------------------
annotations_file = "data/coco/annotations/captions_train2014.json"
output_dir = "data/coco/generated_sd1_4_20"
cfg_scale = 20.0
num_images = 1000
steps = 50  # Default steps

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# -------------------------------
# 3. Load annotations and build mappings
# -------------------------------
print("Loading COCO annotations...")
with open(annotations_file, "r") as f:
    data = json.load(f)

# Mapping from image id to file name
images_data = data["images"]
image_id_to_filename = {img["id"]: img["file_name"] for img in images_data}

# Build mapping from image id to a list of captions
image_id_to_captions = {}
for ann in data["annotations"]:
    image_id = ann["image_id"]
    caption = ann["caption"]
    if image_id not in image_id_to_captions:
        image_id_to_captions[image_id] = []
    image_id_to_captions[image_id].append(caption)

# -------------------------------
# 4. Sort images by file name and process the first 1000
# -------------------------------
sorted_images = sorted(images_data, key=lambda x: x["file_name"])

# Get list of existing valid images
existing_valid_images = {}
if os.path.exists(output_dir):
    for img_file in os.listdir(output_dir):
        if img_file.endswith(('.jpg', '.png')):
            try:
                image_id = int(img_file.split('_')[-1].split('.')[0])
                file_path = os.path.join(output_dir, img_file)
                # Only consider files larger than 1KB as valid
                if os.path.getsize(file_path) > 1024:
                    existing_valid_images[image_id] = file_path
            except (ValueError, IndexError):
                continue

print(f"Found {len(existing_valid_images)} existing valid images")

# Process images
for i, img in enumerate(tqdm(sorted_images[:num_images], desc="Generating images")):
    image_id = img["id"]
    file_name = img["file_name"]
    
    # Skip if image already exists and is valid
    if image_id in existing_valid_images:
        continue

    # Use the first caption for the image
    caption = image_id_to_captions.get(image_id, ["A generic caption."])[0]

    try:
        # Set seed based on image_id for reproducibility
        generator = torch.Generator("cuda").manual_seed(image_id % 10000)
        
        # Generate image
        result = pipe(
            caption,
            guidance_scale=cfg_scale,
            num_inference_steps=steps,
            generator=generator
        )
        
        # Save the generated image
        output_path = os.path.join(output_dir, file_name)
        result["images"][0].save(output_path)
        
        # Verify the saved image
        if os.path.getsize(output_path) < 1024:
            print(f"Warning: Generated image {file_name} is too small")
        else:
            print(f"Successfully saved: {file_name}")
            
    except Exception as e:
        print(f"Error generating image for '{file_name}': {e}")
        continue
    
    # Clear CUDA cache periodically
    if (i + 1) % 50 == 0:
        torch.cuda.empty_cache()

# Final verification
valid_images = sum(
    1 for img_file in os.listdir(output_dir)
    if img_file.endswith(('.jpg', '.png')) and os.path.getsize(os.path.join(output_dir, img_file)) > 1024
)

print(f"\nGeneration complete: {valid_images}/{num_images} valid images in {output_dir}") 