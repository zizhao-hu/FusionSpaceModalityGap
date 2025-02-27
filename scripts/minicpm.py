import os
import json
import math
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer

###############################################
# 1. Define the COCO Caption Dataset
###############################################

class CocoCaptionDataset(Dataset):
    def __init__(self, annotations_file, images_dir, max_samples=None):
        """
        annotations_file: Path to the COCO captions JSON file.
        images_dir: Directory where images are stored.
        max_samples: (Optional) limit number of samples.
        """
        with open(annotations_file, 'r') as f:
            data = json.load(f)
        self.images_dir = images_dir

        # Build a mapping from image ID to file name.
        self.image_id_to_filename = {img_info['id']: img_info['file_name'] for img_info in data['images']}
        
        # Build a list of (image_path, caption) pairs.
        self.samples = []
        for ann in data['annotations']:
            image_id = ann['image_id']
            caption = ann['caption']
            if image_id in self.image_id_to_filename:
                file_name = self.image_id_to_filename[image_id]
                image_path = os.path.join(images_dir, file_name)
                self.samples.append((image_path, caption))
        
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, caption = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        return image, caption

def collate_fn(batch):
    # Batch is a list of tuples (image, caption)
    images, captions = zip(*batch)
    return list(images), list(captions)

###############################################
# 2. Set Up Model, Tokenizer, and Transforms
###############################################

device = "cuda" if torch.cuda.is_available() else "cpu"

# Do not change the model name.
model = AutoModel.from_pretrained(
    "openbmb/MiniCPM-Llama3-V-2_5-int4",
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(
    "openbmb/MiniCPM-Llama3-V-2_5-int4",
    trust_remote_code=True
)

# Define a vision transform (fixed resize for consistency).
vision_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # fixed resolution
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

###############################################
# 3. Helper Functions to Compute Embeddings
###############################################

@torch.no_grad()
def get_language_embedding(caption):
    """
    Compute a language embedding by tokenizing the caption,
    extracting token embeddings, and mean-pooling.
    """
    inputs = tokenizer(caption, return_tensors="pt", truncation=True).to(device)
    token_emb = model.llm.model.embed_tokens(inputs.input_ids)  # shape: (1, seq_len, hidden_dim)
    lang_emb = token_emb.mean(dim=1)  # (1, hidden_dim)
    return lang_emb.squeeze(0)  # (hidden_dim,)

@torch.no_grad()
def get_image_embedding(image):
    """
    Compute an image embedding by processing the image with the vision module (vpm),
    projecting it via the resampler to the language space, and mean-pooling.
    """
    image_tensor = vision_transform(image).unsqueeze(0).to(device)  # (1, 3, H, W)
    image_tensor = image_tensor.half()  # Convert to fp16 to match model weights.
    
    vision_out = model.vpm(image_tensor)  # expects fp16 input
    vision_embedding = vision_out.last_hidden_state  # (1, num_patches, vision_dim)
    
    # Compute target size (number of patches) using floor division.
    H, W = image_tensor.shape[2], image_tensor.shape[3]
    patch_size = model.vpm.patch_size  # e.g., 14
    tgt_size = torch.tensor([[H // patch_size, W // patch_size]], dtype=torch.int32, device=device)
    
    # Project vision embeddings into language space.
    vision_embedding_resampled = model.resampler(vision_embedding, tgt_size)  # (1, num_image_tokens, embed_dim)
    img_emb = vision_embedding_resampled.mean(dim=1)  # (1, embed_dim)
    return img_emb.squeeze(0)  # (embed_dim,)

###############################################
# 4. Collect 100 Paired Embeddings and Save Them
###############################################

paired_lang_embeddings = []
paired_img_embeddings = []
num_pairs = 100
count = 0

annotations_file = "data/annotations/captions_train2014.json"
images_dir = "data/train2014"
dataset = CocoCaptionDataset(annotations_file, images_dir)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_fn)

for batch in dataloader:
    imgs, caps = batch  # each is a list of length 1
    image = imgs[0]
    caption = caps[0]
    
    lang_emb = get_language_embedding(caption)  # (hidden_dim,)
    img_emb = get_image_embedding(image)          # (hidden_dim,)
    
    paired_lang_embeddings.append(lang_emb.cpu().numpy())
    paired_img_embeddings.append(img_emb.cpu().numpy())
    
    count += 1
    if count >= num_pairs:
        break

# Convert lists to numpy arrays.
paired_lang_embeddings = np.array(paired_lang_embeddings)  # shape (100, hidden_dim)
paired_img_embeddings = np.array(paired_img_embeddings)    # shape (100, hidden_dim)

# L2 normalize each embedding.
paired_lang_embeddings = paired_lang_embeddings / np.linalg.norm(paired_lang_embeddings, axis=1, keepdims=True)
paired_img_embeddings = paired_img_embeddings / np.linalg.norm(paired_img_embeddings, axis=1, keepdims=True)

print("Language embedding shape:", paired_lang_embeddings.shape)
print("Image embedding shape:", paired_img_embeddings.shape)

# Save embeddings to a file named {model}_{dataset}_embeddings.npz
save_filename = "MiniCPM-Llama3-V-2_5-int4_coco_embeddings.npz"
np.savez(save_filename, lang=paired_lang_embeddings, img=paired_img_embeddings)
print(f"Saved embeddings to {save_filename}")
