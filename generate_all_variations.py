import os 
import json
import torch
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

torch.backends.cuda.enable_flash_sdp(True)

def setup_pipeline():
    """Set up and return the Stable Diffusion pipeline"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
    )
    pipe = pipe.to(device)
    # Disable safety checker
    pipe.safety_checker = lambda images, clip_input: (images, [False] * len(images))
    return pipe, device

def load_coco_annotations(annotations_file):
    """Load and process COCO annotations"""
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

    return data["images"], image_id_to_captions

def generate_images(pipe, output_dir, sorted_images, image_id_to_captions, cfg_scale, num_steps=50, num_images=1000, batch_size=4):
    """Generate images in batches for a specific configuration"""
    os.makedirs(output_dir, exist_ok=True)

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

    print(f"Found {len(existing_valid_images)} existing valid images in {output_dir}")

    # Process images in batches
    for i in tqdm(range(0, min(len(sorted_images), num_images), batch_size), 
                 desc=f"Generating images for {output_dir}"):
        batch_images = sorted_images[i:min(i + batch_size, num_images)]
        batch_captions = []
        batch_image_ids = []
        batch_file_names = []
        
        # Prepare batch data
        for img in batch_images:
            image_id = img["id"]
            if image_id in existing_valid_images:
                continue
                
            file_name = img["file_name"]
            caption = image_id_to_captions.get(image_id, ["A generic caption."])[0]
            
            batch_captions.append(caption)
            batch_image_ids.append(image_id)
            batch_file_names.append(file_name)
        
        if not batch_captions:  # Skip if no images to generate in this batch
            continue

        try:
            # Generate seeds for the batch
            generators = [torch.Generator("cuda").manual_seed(image_id % 10000) 
                        for image_id in batch_image_ids]
            
            # Generate images for the batch
            results = pipe(
                batch_captions,
                guidance_scale=cfg_scale,
                num_inference_steps=num_steps,
                generator=generators
            )
            
            # Save generated images
            for idx, (image, file_name) in enumerate(zip(results["images"], batch_file_names)):
                output_path = os.path.join(output_dir, file_name)
                image.save(output_path)
                
                # Verify the saved image
                if os.path.getsize(output_path) < 1024:
                    print(f"Warning: Generated image {file_name} is too small")
                else:
                    print(f"Successfully saved: {file_name}")
                
        except Exception as e:
            print(f"Error generating batch: {str(e)}")
            continue
        
        # Clear CUDA cache periodically
        if (i + batch_size) % 50 == 0:
            torch.cuda.empty_cache()

    # Final verification
    valid_images = sum(
        1 for img_file in os.listdir(output_dir)
        if img_file.endswith(('.jpg', '.png')) and os.path.getsize(os.path.join(output_dir, img_file)) > 1024
    )

    print(f"\nGeneration complete for {output_dir}: {valid_images}/{num_images} valid images")

def main():
    # Set up paths and parameters
    base_dir = "data/coco"
    annotations_file = os.path.join(base_dir, "annotations/captions_train2014.json")
    num_images = 1000
    default_steps = 50
    batch_size = 10

    # Set up configurations to generate
    cfg_variations = [
        {"name": "generated_sd1_4_1", "cfg": 1.0, "steps": default_steps},
        {"name": "generated_sd1_4_3", "cfg": 3.0, "steps": default_steps},
        {"name": "generated_sd1_4_7", "cfg": 7.0, "steps": default_steps},
        {"name": "generated_sd1_4_10", "cfg": 10.0, "steps": default_steps},
        {"name": "generated_sd1_4_20", "cfg": 20.0, "steps": default_steps},
    ]

    step_variations = [
        {"name": "generated_sd1_4_7_steps_10", "cfg": 7.0, "steps": 10},
        {"name": "generated_sd1_4_7_steps_20", "cfg": 7.0, "steps": 20},
        {"name": "generated_sd1_4_7_steps_50", "cfg": 7.0, "steps": 50},
        {"name": "generated_sd1_4_7_steps_100", "cfg": 7.0, "steps": 100},
        {"name": "generated_sd1_4_7_steps_200", "cfg": 7.0, "steps": 200},
        {"name": "generated_sd1_4_7_steps_500", "cfg": 7.0, "steps": 500},
    ]

    # Set up pipeline
    pipe, device = setup_pipeline()

    # Load COCO annotations
    images_data, image_id_to_captions = load_coco_annotations(annotations_file)
    sorted_images = sorted(images_data, key=lambda x: x["file_name"])

    # Generate images for CFG variations
    print("\nGenerating CFG variations...")
    for config in cfg_variations:
        output_dir = os.path.join(base_dir, config["name"])
        print(f"\nProcessing {config['name']} (CFG={config['cfg']}, Steps={config['steps']})")
        generate_images(pipe, output_dir, sorted_images, image_id_to_captions, 
                       config["cfg"], config["steps"], num_images, batch_size)

    # Generate images for step variations
    print("\nGenerating step variations...")
    for config in step_variations:
        output_dir = os.path.join(base_dir, config["name"])
        print(f"\nProcessing {config['name']} (CFG={config['cfg']}, Steps={config['steps']})")
        generate_images(pipe, output_dir, sorted_images, image_id_to_captions, 
                       config["cfg"], config["steps"], num_images, batch_size)

if __name__ == "__main__":
    main() 