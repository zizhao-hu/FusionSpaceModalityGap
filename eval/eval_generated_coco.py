import os
import numpy as np
import torch
import torchvision.transforms as T
from glob import glob
from PIL import Image
from tqdm import tqdm
import torchvision.models as models
import torch.nn.functional as F
import scipy.linalg
import scipy.stats
from pytorch_fid import fid_score
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"

###############################################
# Settings & File Lists (only first 100 images)
###############################################
real_folder = "data/train2014"  # real images folder
# Get first 100 real image file paths.
real_files = sorted(glob(os.path.join(real_folder, "*.jpg")) + 
                    glob(os.path.join(real_folder, "*.png")))[:100]

# Define generated folders for each CFG.
gen_folders = {
    "CFG 1": "data/generated_sd1_4_1",
    "CFG 3": "data/generated_sd1_4_3",
    "CFG 7": "data/generated_sd1_4_7",
    "CFG 10": "data/generated_sd1_4_10",
    "CFG 20": "data/generated_sd1_4_20",
}

###############################################
# Transform for Inception (for FID & IS)
###############################################
inception_transform = T.Compose([
    T.Resize((299, 299)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

###############################################
# Custom Dataset using transform (for FID & IS)
###############################################
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, files, transform):
        self.files = files
        self.transform = transform
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        try:
            img = Image.open(self.files[idx]).convert("RGB")
        except Exception as e:
            print(f"Error opening {self.files[idx]}: {e}")
            raise e
        return self.transform(img)

###############################################
# Prepare Inception v3 for FID & IS
###############################################
inception_model = models.inception_v3(pretrained=True, transform_input=False).to(device)
inception_model.eval()

###############################################
# FID Calculation using a custom DataLoader with transform
###############################################
def compute_fid(real_files, gen_files, batch_size=16, device=device, dims=2048):
    # Real images activations.
    dataset_real = ImageDataset(real_files, inception_transform)
    loader_real = torch.utils.data.DataLoader(dataset_real, batch_size=batch_size, shuffle=False, num_workers=0)
    real_acts = []
    with torch.no_grad():
        for batch in tqdm(loader_real, desc="Real Images"):
            batch = batch.to(device)
            pred = inception_model(batch)
            real_acts.append(pred.cpu().numpy())
    real_acts = np.concatenate(real_acts, axis=0)
    mu_real = np.mean(real_acts, axis=0)
    sigma_real = np.cov(real_acts, rowvar=False)
    
    # Generated images activations.
    dataset_gen = ImageDataset(gen_files, inception_transform)
    loader_gen = torch.utils.data.DataLoader(dataset_gen, batch_size=batch_size, shuffle=False, num_workers=0)
    gen_acts = []
    with torch.no_grad():
        for batch in tqdm(loader_gen, desc="Gen Images"):
            batch = batch.to(device)
            pred = inception_model(batch)
            gen_acts.append(pred.cpu().numpy())
    gen_acts = np.concatenate(gen_acts, axis=0)
    mu_gen = np.mean(gen_acts, axis=0)
    sigma_gen = np.cov(gen_acts, rowvar=False)
    
    diff = mu_real - mu_gen
    covmean, _ = scipy.linalg.sqrtm(sigma_real.dot(sigma_gen), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid_val = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2*covmean)
    return fid_val

###############################################
# Inception Score (IS) Calculation
###############################################
def compute_inception_score(files, batch_size=16, splits=10, device=device):
    dataset = ImageDataset(files, inception_transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing IS"):
            batch = batch.to(device)
            pred = inception_model(batch)
            preds.append(F.softmax(pred, dim=1))
    preds = torch.cat(preds, dim=0).cpu().numpy()
    N = preds.shape[0]
    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits):(k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = [scipy.stats.entropy(part[i], py) for i in range(part.shape[0])]
        split_scores.append(np.exp(np.mean(scores)))
    return np.mean(split_scores), np.std(split_scores)

###############################################
# CLIP Score Calculation
###############################################
# Load caption and file name arrays from a NPZ file (generated from CLIP embeddings for CFG=1)
cap_npz_path = os.path.join("data", "CLIP_openai_clip-vit-base-patch32_embeddings_cfg_1.npz")
if not os.path.exists(cap_npz_path):
    raise FileNotFoundError(f"Caption NPZ file not found: {cap_npz_path}")
cap_data = np.load(cap_npz_path)
captions_list = np.array(cap_data["captions"])  # shape (N,)
file_names = np.array(cap_data["file_names"])   # shape (N,)

def compute_clip_score(files, captions_list, file_names, gen_folder, device=device):
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_model.eval()
    scores = []
    for f in files:
        fname = os.path.basename(f)
        idx = np.where(file_names == fname)[0]
        if len(idx) == 0:
            continue
        caption = captions_list[idx[0]]
        gen_path = os.path.join(gen_folder, fname)
        if not os.path.exists(gen_path):
            continue
        try:
            image = Image.open(gen_path).convert("RGB")
        except Exception as e:
            print(f"Error opening {gen_path}: {e}")
            continue
        inputs = clip_processor(text=[caption], images=image, return_tensors="pt", padding=True)
        image_inputs = {"pixel_values": inputs["pixel_values"].to(device)}
        text_inputs = {"input_ids": inputs["input_ids"].to(device),
                       "attention_mask": inputs["attention_mask"].to(device)}
        with torch.no_grad():
            image_features = clip_model.get_image_features(**image_inputs)
            text_features = clip_model.get_text_features(**text_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        score = torch.cosine_similarity(image_features, text_features).item()
        scores.append(score)
    return np.mean(scores) if scores else 0

###############################################
# Main Evaluation Loop for All Generated Folders
###############################################
if __name__ == "__main__":
    print("Evaluating Generated Images Quality Metrics (100 images each):\n")
    
    # Loop over each generated folder.
    for cfg_label, gen_folder in gen_folders.items():
        # Get first 100 generated files from this folder.
        gen_files = sorted(glob(os.path.join(gen_folder, "*.jpg")) + glob(os.path.join(gen_folder, "*.png")))[:100]
        if len(gen_files) == 0:
            print(f"No generated files found in {gen_folder}, skipping {cfg_label}.")
            continue
        
        print(f"--- {cfg_label} ---")
        
        fid_val = compute_fid(real_files, gen_files, batch_size=16, device=device, dims=2048)
        print(f"{cfg_label} FID: {fid_val:.2f}")
        
        is_mean, is_std = compute_inception_score(gen_files, batch_size=16, splits=10, device=device)
        print(f"{cfg_label} Inception Score: {is_mean:.2f} ± {is_std:.2f}")
        
        clip_score = compute_clip_score(real_files, captions_list, file_names, gen_folder, device=device)
        print(f"{cfg_label} Average CLIP Score: {clip_score:.3f}\n")
