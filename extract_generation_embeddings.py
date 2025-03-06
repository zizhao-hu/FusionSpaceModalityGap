import os
import json
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# Set up paths
data_root = "data"
output_folder = os.path.join(data_root, "embeddings")
os.makedirs(output_folder, exist_ok=True)

# Load CLIP model for embedding extraction
model_name = "openai/clip-vit-base-patch32"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

def load_coco_captions(annotation_file=None):
    """Load captions from COCO dataset, getting the first caption for each image."""
    if annotation_file is None:
        annotation_file = os.path.join(data_root, "coco", "annotations", "captions_train2014.json")
    
    if not os.path.exists(annotation_file):
        print(f"COCO annotation file not found: {annotation_file}")
        # Return a dummy caption if file doesn't exist
        return ["A photograph of an object."] * 100, {}
    
    print(f"Loading captions from {annotation_file}")
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    # Group captions by image_id
    captions_by_image = {}
    for ann in data['annotations']:
        image_id = ann['image_id']
        if image_id not in captions_by_image:
            captions_by_image[image_id] = []
        captions_by_image[image_id].append(ann['caption'])
    
    # Get the first caption for each image
    first_captions = {}
    for image_id, captions in captions_by_image.items():
        first_captions[image_id] = captions[0]
    
    # Get the first 100 unique image IDs
    image_ids = list(first_captions.keys())[:100]
    
    # Create a list of the first 100 captions (one per image)
    captions = [first_captions[img_id] for img_id in image_ids]
    
    print(f"Loaded {len(captions)} captions from COCO dataset (first caption per image)")
    
    # Create a mapping from index to caption for later use
    caption_mapping = {i: caption for i, caption in enumerate(captions)}
    
    return captions, caption_mapping

def extract_embeddings(image_path, caption):
    """Extract CLIP embeddings for an image and its caption."""
    try:
        # Load and process the image
        image = Image.open(image_path).convert("RGB")
        
        # Process inputs
        inputs = processor(
            text=[caption],
            images=image,
            return_tensors="pt",
            padding=True
        ).to(device)
        
        # Extract embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Get normalized embeddings
            image_embeds = outputs.image_embeds.cpu().numpy()
            text_embeds = outputs.text_embeds.cpu().numpy()
            
        return text_embeds[0], image_embeds[0]
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None

def process_generation(gen_number, caption_mapping, max_samples=100):
    """Process all images for a specific generation using their corresponding captions."""
    # Define paths for this generation using the correct folder naming convention
    gen_dir = os.path.join(data_root, "coco", f"sd_to_sd_cfg_7_steps_50_gen_{gen_number}")
    
    # Check if the directory exists
    if not os.path.exists(gen_dir):
        print(f"Generation directory not found: {gen_dir}")
        return None, None
    
    # Get list of image files
    image_files = [f for f in os.listdir(gen_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Sort the image files to ensure they match the caption indices
    image_files.sort()
    
    # Limit the number of samples if needed
    image_files = image_files[:max_samples]
    
    # Initialize arrays to store embeddings and captions used
    text_embeddings = []
    image_embeddings = []
    captions_used = []
    
    # Process each image with its corresponding caption
    for idx, img_file in enumerate(tqdm(image_files, desc=f"Processing Gen {gen_number}")):
        if idx >= len(caption_mapping):
            break
            
        img_path = os.path.join(gen_dir, img_file)
        caption = caption_mapping[idx]
        
        # Extract embeddings using the corresponding caption
        text_emb, img_emb = extract_embeddings(img_path, caption)
        
        if text_emb is not None and img_emb is not None:
            text_embeddings.append(text_emb)
            image_embeddings.append(img_emb)
            captions_used.append(caption)
    
    # Convert to numpy arrays
    text_embeddings = np.array(text_embeddings)
    image_embeddings = np.array(image_embeddings)
    
    print(f"Processed {len(text_embeddings)} samples for generation {gen_number}")
    
    return text_embeddings, image_embeddings, captions_used

def process_all_captions(captions):
    """Process all captions to get their embeddings."""
    text_embeddings = []
    
    print("Processing all captions...")
    for caption in tqdm(captions):
        # Process caption only
        inputs = processor(
            text=[caption],
            return_tensors="pt",
            padding=True
        ).to(device)
        
        # Extract embeddings
        with torch.no_grad():
            outputs = model.get_text_features(**inputs)
            text_emb = outputs.cpu().numpy()
        
        text_embeddings.append(text_emb[0])
    
    # Convert to numpy array
    text_embeddings = np.array(text_embeddings)
    print(f"Processed {len(text_embeddings)} captions")
    
    return text_embeddings

# Load COCO captions and create mapping
coco_captions, caption_mapping = load_coco_captions()

# Process all captions to get their embeddings
caption_embeddings = process_all_captions(coco_captions)

# Save caption embeddings
caption_output_path = os.path.join(output_folder, "coco_caption_embeddings.npz")
np.savez(
    caption_output_path,
    text_embeddings=caption_embeddings,
    captions=coco_captions
)
print(f"Saved caption embeddings to {caption_output_path}")

# Process generations 0, 4, and 10
generations = [0, 4, 10]

for gen in generations:
    print(f"\nExtracting embeddings for generation {gen}...")
    text_emb, img_emb, captions_used = process_generation(gen, caption_mapping)
    
    if text_emb is not None and img_emb is not None:
        # Save embeddings
        output_path = os.path.join(output_folder, f"gen_{gen}_embeddings.npz")
        np.savez(
            output_path,
            text_embeddings=text_emb,
            image_embeddings=img_emb,
            captions=captions_used  # Save all captions used
        )
        print(f"Saved embeddings to {output_path}")
        print(f"Text embeddings shape: {text_emb.shape}")
        print(f"Image embeddings shape: {img_emb.shape}")

print("\nEmbedding extraction complete!") 