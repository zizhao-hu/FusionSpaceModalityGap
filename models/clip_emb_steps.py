import os
import json
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

###############################################
# 1. Build mapping from COCO annotations for first 100 images
###############################################
annotations_file = "data/coco/annotations/captions_train2014.json"
original_images_dir = "data/coco/train2014"  # used only for file names

with open(annotations_file, 'r') as f:
    data = json.load(f)

# Get the list of images from the annotations JSON and sort them by file name.
images_data = data["images"]
sorted_images = sorted(images_data, key=lambda x: x["file_name"])

# Build a mapping from image id to file name.
image_id_to_filename = {img["id"]: img["file_name"] for img in images_data}

# Build a mapping from image id to its list of captions.
image_id_to_captions = {}
for ann in data["annotations"]:
    image_id = ann["image_id"]
    caption = ann["caption"]
    if image_id not in image_id_to_captions:
        image_id_to_captions[image_id] = []
    image_id_to_captions[image_id].append(caption)

# For the first 100 images (sorted by file name), build a mapping from file name to caption.
num_images = 100
file_to_caption = {}
for img in sorted_images[:num_images]:
    img_id = img["id"]
    file_name = img["file_name"]
    if img_id in image_id_to_captions:
        file_to_caption[file_name] = image_id_to_captions[img_id][0]
    else:
        file_to_caption[file_name] = "A generic caption."

###############################################
# 2. Set Up CLIP Model and Processor (CLS embeddings only)
###############################################
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(model_name, output_hidden_states=True).to(device)
clip_processor = CLIPProcessor.from_pretrained(model_name)
clip_model.eval()

TEXT_MAX_LENGTH = 77

@torch.no_grad()
def get_clip_cls_text_embedding(caption: str):
    inputs = clip_processor(
        text=[caption],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=TEXT_MAX_LENGTH
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    text_emb = clip_model.get_text_features(**inputs)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    return text_emb.squeeze(0)  # shape: (512,)

@torch.no_grad()
def get_clip_cls_image_embedding(image: Image.Image):
    inputs = clip_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    image_emb = clip_model.get_image_features(**inputs)
    image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
    return image_emb.squeeze(0)  # shape: (512,)

###############################################
# 3. Process and Save Generated Images Embeddings for scale 7 with sampling steps 20, 100, 200, 500
###############################################
cfg = 7
sampling_steps_list = [10, 20, 100, 200, 500]
base_folder_prefix = "data/coco/generated_sd1_4_"

for steps in sampling_steps_list:
    folder_name = f"{base_folder_prefix}{cfg}_steps_{steps}"
    if not os.path.exists(folder_name):
        print(f"Folder {folder_name} does not exist. Skipping scale {cfg} with steps {steps}.")
        continue

    image_embeddings = []
    text_embeddings = []
    captions_list = []
    file_names_list = []

    for file_name in sorted(os.listdir(folder_name)):
        if not (file_name.lower().endswith(".jpg") or file_name.lower().endswith(".png")):
            continue
        image_path = os.path.join(folder_name, file_name)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            continue

        caption = file_to_caption.get(file_name, "Caption not found.")
        print(f"[Generated Scale {cfg} Steps {steps}] Processing {file_name}: {caption}")

        img_emb = get_clip_cls_image_embedding(image)
        txt_emb = get_clip_cls_text_embedding(caption)

        image_embeddings.append(img_emb.cpu().numpy())
        text_embeddings.append(txt_emb.cpu().numpy())
        captions_list.append(caption)
        file_names_list.append(file_name)

    image_embeddings = np.array(image_embeddings)
    text_embeddings = np.array(text_embeddings)
    captions_array = np.array(captions_list)
    file_names_array = np.array(file_names_list)

    save_filename = f"CLIP_{model_name.replace('/', '_')}_embeddings_cfg_{cfg}_steps_{steps}.npz"
    np.savez(save_filename,
             image_embeddings=image_embeddings,
             text_embeddings=text_embeddings,
             captions=captions_array,
             file_names=file_names_array)
    print(f"Saved generated embeddings for Scale {cfg} Steps {steps} to {save_filename}\n")

###############################################
# 4. Process and Save Real Images Embeddings (first 100 images)
###############################################
real_image_embeddings = []
real_text_embeddings = []
real_captions_list = []
real_file_names = []

for file_name in sorted(os.listdir(original_images_dir))[:num_images]:
    if not (file_name.lower().endswith(".jpg") or file_name.lower().endswith(".png")):
        continue
    image_path = os.path.join(original_images_dir, file_name)
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening real image {image_path}: {e}")
        continue

    caption = file_to_caption.get(file_name, "Caption not found.")
    print(f"[Real] Processing {file_name}: {caption}")

    img_emb = get_clip_cls_image_embedding(image)
    txt_emb = get_clip_cls_text_embedding(caption)

    real_image_embeddings.append(img_emb.cpu().numpy())
    real_text_embeddings.append(txt_emb.cpu().numpy())
    real_captions_list.append(caption)
    real_file_names.append(file_name)

real_image_embeddings = np.array(real_image_embeddings)
real_text_embeddings = np.array(real_text_embeddings)
real_captions_array = np.array(real_captions_list)
real_file_names_array = np.array(real_file_names)

save_real_filename = f"CLIP_{model_name.replace('/', '_')}_embeddings_real.npz"
np.savez(save_real_filename,
         image_embeddings=real_image_embeddings,
         text_embeddings=real_text_embeddings,
         captions=real_captions_array,
         file_names=real_file_names_array)
print(f"Saved real images embeddings to {save_real_filename}")
