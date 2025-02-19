import os
import json
import torch
from diffusers import StableDiffusionPipeline

# -------------------------------
# 1. Set up device and load Stable Diffusion pipeline
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16,
)
pipe = pipe.to(device)
# Disable safety checker by returning one flag per image.
pipe.safety_checker = lambda images, clip_input: (images, [False] * len(images))

# -------------------------------
# 2. Define paths and parameters
# -------------------------------
annotations_file = "data/annotations/captions_train2014.json"
# We'll use the "images" info for naming.
model_version = "sd1_4"
cfg_scales = [1, 3, 7, 10, 20]
# Create an output folder for each CFG scale.
output_dirs = {}
for cfg in cfg_scales:
    folder_name = f"generated_{model_version}_{cfg}"
    os.makedirs(folder_name, exist_ok=True)
    output_dirs[cfg] = folder_name

# -------------------------------
# 3. Load annotations and build mappings
# -------------------------------
with open(annotations_file, "r") as f:
    data = json.load(f)

# Mapping from image id to file name (from the "images" field)
images_data = data["images"]
image_id_to_filename = {img["id"]: img["file_name"] for img in images_data}

# Build mapping from image id to a list of captions.
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
num_images = 1000  # first 100 images

for i, img in enumerate(sorted_images[:num_images]):
    image_id = img["id"]
    file_name = img["file_name"]

    # Use the first caption for the image if available; otherwise, use a fallback.
    if image_id in image_id_to_captions:
        caption = image_id_to_captions[image_id][0]
    else:
        caption = "A generic caption."

    # Optionally, print progress.
    print(f"Processing image {i+1}/{num_images}: {file_name} with caption: {caption}")

    # Generate an image for each CFG scale.
    for cfg in cfg_scales:
        try:
            result = pipe(caption, guidance_scale=cfg)
            # In diffusers v0.13+, the generated images are returned in result["images"]
            generated_image = result["images"][0]
        except Exception as e:
            print(f"Error generating image for '{file_name}' with cfg {cfg}: {e}")
            continue

        # Save the generated image using the original file name.
        output_path = os.path.join(output_dirs[cfg], file_name)
        try:
            generated_image.save(output_path)
            print(f"Saved: {output_path}")
        except Exception as e:
            print(f"Error saving image {file_name} with cfg {cfg}: {e}")
