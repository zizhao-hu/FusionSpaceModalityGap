import os
import numpy as np
import matplotlib.pyplot as plt
import umap
import sys
import os

# Compute the repository root (one level up from the 'scripts' directory)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
    
from utils.rmg import rmg_cosine_dissimilarity

# ----- Parameters & Paths -----
data_folder = "data"
model_fname = "openai_clip-vit-base-patch32"  # model name with "/" replaced by "_"
num_samples = 100

# NPZ for real images (pre-saved from train2014)
real_npz = os.path.join(data_folder, f"CLIP_{model_fname}_embeddings_real.npz")
# Generated folders NPZ files for CFG scales 1, 3, 7, 10, 20
cfg_scales = [1, 3, 7, 10, 20]

# Colors: captions in blue, images in red.
cap_color, img_color = "blue", "red"

# ----- Load Real Embeddings -----
real_data = np.load(real_npz)
real_text_emb = np.array(real_data["text_embeddings"])  # (100, 512)
real_img_emb = np.array(real_data["image_embeddings"])   # (100, 512)
real_caps = np.array(real_data["captions"])              # (100,)
real_files = np.array(real_data["file_names"])           # (100,)

# ----- Function to Compute Metrics -----
def compute_metrics(cap, img):
    cos = np.mean(np.sum(cap * img, axis=1))
    l2m = np.linalg.norm(np.mean(cap, axis=0) - np.mean(img, axis=0))
    rmg = rmg_cosine_dissimilarity(cap, img)
    return cos, l2m, rmg

# ----- Create Subplots -----
# Total subplots: 1 (real) + 5 (for each CFG) = 6
fig, axes = plt.subplots(1, len(cfg_scales) + 1, figsize=(36, 6))

# ----- Subplot 0: Real Images vs. Captions -----
combined_real = np.concatenate([real_text_emb, real_img_emb], axis=0)  # (200, 512)
reducer = umap.UMAP(n_components=2, random_state=42)
real_2d = reducer.fit_transform(combined_real)
real_cap_2d, real_img_2d = real_2d[:num_samples, :], real_2d[num_samples:, :]

avg_cos, l2m, rmg = compute_metrics(real_text_emb, real_img_emb)
ax = axes[0]
ax.scatter(real_cap_2d[:, 0], real_cap_2d[:, 1], c=cap_color, s=50, marker="o", label="Caption")
ax.scatter(real_img_2d[:, 0], real_img_2d[:, 1], c=img_color, s=50, marker="s", label="Real Image")
for j in range(num_samples):
    ax.plot([real_cap_2d[j, 0], real_img_2d[j, 0]],
            [real_cap_2d[j, 1], real_img_2d[j, 1]],
            c=img_color, lw=0.5, alpha=0.7)
ax.set_title(f"Real\n$\mathbf{{COS}}$: {avg_cos:.3f}  $\mathbf{{L2M}}$: {l2m:.3f}  $\mathbf{{RMG}}$: {rmg:.3f}", fontsize=18)
ax.set_xlabel("UMAP Dim 1", fontsize=16, fontweight="bold")
ax.set_ylabel("UMAP Dim 2", fontsize=16, fontweight="bold")
ax.legend(fontsize=14)

# ----- Subplots 1-5: Generated Images (for each CFG) vs. Real Captions -----
base_npz_prefix = os.path.join(data_folder, f"CLIP_{model_fname}_embeddings_cfg_")
for idx, cfg in enumerate(cfg_scales, start=1):
    gen_npz = f"{base_npz_prefix}_{cfg}.npz".replace("__", "_")  # ensure correct naming
    if not os.path.exists(gen_npz):
        print(f"File not found: {gen_npz}. Skipping CFG {cfg}.")
        continue
    gen_data = np.load(gen_npz)
    gen_img_emb = np.array(gen_data["image_embeddings"])  # (100, 512)
    # Use the real captions (real_text_emb) for comparison.
    avg_cos, l2m, rmg = compute_metrics(real_text_emb, gen_img_emb)
    combined = np.concatenate([real_text_emb, gen_img_emb], axis=0)  # (200, 512)
    reducer = umap.UMAP(n_components=2, random_state=42)
    pair_2d = reducer.fit_transform(combined)
    cap_2d, img_2d = pair_2d[:num_samples, :], pair_2d[num_samples:, :]
    ax = axes[idx]
    ax.scatter(cap_2d[:, 0], cap_2d[:, 1], c=cap_color, s=50, marker="o")
    ax.scatter(img_2d[:, 0], img_2d[:, 1], c=img_color, s=50, marker="s")
    for j in range(num_samples):
        ax.plot([cap_2d[j, 0], img_2d[j, 0]], [cap_2d[j, 1], img_2d[j, 1]], c=img_color, lw=0.5, alpha=0.7)
    ax.set_title(f"CFG {cfg}\n$\mathbf{{COS}}$: {avg_cos:.3f}  $\mathbf{{L2M}}$: {l2m:.3f}  $\mathbf{{RMG}}$: {rmg:.3f}", fontsize=18)
    # Remove x and y tick labels (no x/y axis titles or legends)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

plt.tight_layout(rect=[0, 0, 1, 0.95])
output_plot = os.path.join(data_folder, "combined_individual_umap_plot_with_real.png")
plt.savefig(output_plot, dpi=300)
plt.close()
print(f"Saved plot to {output_plot}")
