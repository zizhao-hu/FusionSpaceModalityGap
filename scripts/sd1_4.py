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
annotations_file = "data/coco/annotations/captions_train2014.json"
model_version = "sd1_4"
cfg_scales = [7]
sampling_steps_list = [10]  # Default is 50

# Build output directories for each combination (cfg, sampling steps)
# For steps==50, use the default naming convention (without adding steps)
output_dirs = {}
# We'll also mark which (cfg, steps) combinations have already been generated.
skip_combinations = set()

for cfg in cfg_scales:
    for steps in sampling_steps_list:
        if steps == 50:
            folder_name = f"generated_{model_version}_{cfg}"
        else:
            folder_name = f"generated_{model_version}_{cfg}_steps_{steps}"
        # Check if folder already exists and is non-empty (i.e. already generated)
        if os.path.exists(folder_name) and os.listdir(folder_name):
            print(f"Folder {folder_name} already exists and is non-empty. Skipping generation for cfg {cfg} with steps {steps}.")
            skip_combinations.add((cfg, steps))
        else:
            os.makedirs(folder_name, exist_ok=True)
        output_dirs[(cfg, steps)] = folder_name

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
num_images = 100

for i, img in enumerate(sorted_images[:num_images]):
    image_id = img["id"]
    file_name = img["file_name"]

    # Use the first caption for the image if available; otherwise, use a fallback.
    caption = image_id_to_captions.get(image_id, ["A generic caption."])[0]

    print(f"Processing image {i+1}/{num_images}: {file_name} with caption: {caption}")

    # Generate an image for each combination of CFG scale and sampling steps.
    for cfg in cfg_scales:
        for steps in sampling_steps_list:
            if (cfg, steps) in skip_combinations:
                continue
            try:
                result = pipe(caption, guidance_scale=cfg, num_inference_steps=steps)
                # For diffusers v0.13+, images are in result["images"]
                generated_image = result["images"][0]
            except Exception as e:
                print(f"Error generating image for '{file_name}' with cfg {cfg} and steps {steps}: {e}")
                continue

            # Save the generated image using the original file name.
            output_path = os.path.join(output_dirs[(cfg, steps)], file_name)
            try:
                generated_image.save(output_path)
                print(f"Saved: {output_path}")
            except Exception as e:
                print(f"Error saving image {file_name} with cfg {cfg} and steps {steps}: {e}")
