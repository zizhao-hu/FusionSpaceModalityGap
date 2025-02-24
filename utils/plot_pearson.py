import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ----- Parameters & Paths -----
data_folder = "data"
model_fname = "openai_clip-vit-base-patch32"  # model name with "/" replaced by "_"
num_samples = 100
real_npz = os.path.join(data_folder, f"vis\CLIP_{model_fname}_embeddings_real.npz")
cfg_scales = [1, 3, 7, 10, 20]

# ----- Load Real Embeddings -----
real_data = np.load(real_npz)
real_text_emb = np.array(real_data["text_embeddings"])  # shape: (100, 512)
real_img_emb = np.array(real_data["image_embeddings"])    # shape: (100, 512)

# ----- Function to Compute Per-Pair Metrics (Cosine and L2) -----
def compute_pairwise_metrics(cap, img):
    """
    Computes two metrics per image-caption pair:
      - Cosine similarity (via dot product; assumes embeddings are normalized)
      - L2 distance between the embeddings
    """
    # Compute cosine similarity per pair
    cos_vals = np.sum(cap * img, axis=1)  # shape: (num_pairs,)
    # Compute L2 distance per pair
    l2_vals = np.linalg.norm(cap - img, axis=1)
    return cos_vals, l2_vals

# ----- Gather Metrics Across Conditions -----
all_cos, all_l2 = [], []

# Process real pairs
cos_vals, l2_vals = compute_pairwise_metrics(real_text_emb, real_img_emb)
all_cos.append(cos_vals)
all_l2.append(l2_vals)

# Process generated pairs for each CFG scale (using real text embeddings for captions)
base_npz_prefix = os.path.join(data_folder, f"vis\CLIP_{model_fname}_embeddings_cfg_")
for cfg in cfg_scales:
    gen_npz = f"{base_npz_prefix}_{cfg}.npz".replace("__", "_")  # adjust naming if needed
    if not os.path.exists(gen_npz):
        print(f"File not found: {gen_npz}. Skipping CFG {cfg}.")
        continue
    gen_data = np.load(gen_npz)
    gen_img_emb = np.array(gen_data["image_embeddings"])  # shape: (100, 512)
    cos_vals, l2_vals = compute_pairwise_metrics(real_text_emb, gen_img_emb)
    all_cos.append(cos_vals)
    all_l2.append(l2_vals)

# Concatenate all metrics (each array from a condition has shape (100,))
all_cos = np.concatenate(all_cos, axis=0)  # combined ~600 values
all_l2 = np.concatenate(all_l2, axis=0)

# ----- Compute Pearson Correlation -----
r_value, p_value = pearsonr(all_cos, all_l2)

# ----- Plotting the Scatter Plot -----
plt.figure(figsize=(8, 6))
plt.scatter(all_cos, all_l2, alpha=0.7, s=50, color='purple')
plt.xlabel("Cosine Similarity", fontsize=14, fontweight="bold")
plt.ylabel("L2 Distance", fontsize=14, fontweight="bold")
plt.title(f"Cosine Similarity vs L2 Distance\nPearson r: {r_value:.3f} (p={p_value:.2e})", fontsize=16)
plt.tight_layout()

output_plot = os.path.join(data_folder, "cosine_vs_l2_scatter_plot_all600_pairs.png")
plt.savefig(output_plot, dpi=300)
plt.close()
print(f"Saved scatter plot to {output_plot}")
