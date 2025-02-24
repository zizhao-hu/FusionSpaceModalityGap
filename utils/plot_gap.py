import os, sys, numpy as np, pandas as pd, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D

def rmg_cosine_dissimilarity(x, y):
    """
    Compute the RMG value from the given formula, but using
    cosine dissimilarity instead of Euclidean distance.
    
    RMG =  [ (1/N) * sum_i d(x_i, y_i) ]  /
           [ (1/(2*N*(N-1))) * ( sum_{i!=j} d(x_i, x_j) + sum_{i!=j} d(y_i, y_j) )
             + (1/N) * sum_i d(x_i, y_i ) ]
    
    where d(a, b) = 1 - (a·b)/(||a|| * ||b||).
    """
    N = x.shape[0]
    
    def cosine_dissim_rowwise(A, B):
        numerator = np.einsum('ij,ij->i', A, B)
        normA = np.linalg.norm(A, axis=1)
        normB = np.linalg.norm(B, axis=1)
        cos_sim = numerator / (normA * normB)
        return 1.0 - cos_sim  # dissimilarity
    
    row_dissim_xy = cosine_dissim_rowwise(x, y)
    numerator = np.mean(row_dissim_xy)
    
    def sum_pairwise_cos_dissim(M):
        dot_mat = M @ M.T
        norms = np.linalg.norm(M, axis=1)
        norm_mat = np.outer(norms, norms)
        cos_sim_mat = dot_mat / norm_mat
        cos_dissim_mat = 1.0 - cos_sim_mat
        return np.sum(cos_dissim_mat)
    
    sum_dxx = sum_pairwise_cos_dissim(x)
    sum_dyy = sum_pairwise_cos_dissim(y)
    denom_part1 = (1.0 / (2.0 * N * (N - 1))) * (sum_dxx + sum_dyy)
    denom_part2 = numerator
    denominator = denom_part1 + denom_part2
    rmg_value = numerator / denominator
    return rmg_value

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Setup paths/parameters – note the data folder is now "data/vis"
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
data_folder = os.path.join("data", "vis")
model_fname = "openai_clip-vit-base-patch32"
num_samples = 100

# Load real data
real_npz = os.path.join(data_folder, f"CLIP_{model_fname}_embeddings_real.npz")
real_data = np.load(real_npz)
real_text_emb = np.array(real_data["text_embeddings"])
real_img_emb = np.array(real_data["image_embeddings"])

# Updated compute_metrics to use cosine dissimilarity instead
def compute_metrics(cap, img):
    # Compute row-wise cosine similarity for each pair
    row_cos = np.array([cosine_similarity(cap[i], img[i]) for i in range(cap.shape[0])])
    cos_dissim = np.mean(1 - row_cos)  # average cosine dissimilarity
    l2m = np.linalg.norm(np.mean(cap, axis=0) - np.mean(img, axis=0))
    rmg = rmg_cosine_dissimilarity(cap, img)
    return cos_dissim, l2m, rmg

def set_axes_equal(ax):
    ax.set_box_aspect([1,1,1])
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_visible(False)
        axis.pane.set_edgecolor('none')

def hide_axis_lines(ax):
    ax.xaxis.line.set_color((0,0,0,0))
    ax.yaxis.line.set_color((0,0,0,0))
    ax.zaxis.line.set_color((0,0,0,0))

def plot_unit_sphere(ax):
    u, v = np.meshgrid(np.linspace(0, 2*np.pi, 30), np.linspace(0, np.pi, 15))
    x, y, z = np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color='whitesmoke', alpha=0.15,
                     edgecolor='grey', linewidth=0.5, antialiased=True)

def draw_coordinate_arrows(ax, length=1.5):
    ax.quiver(0,0,0, length,0,0, color='black', arrow_length_ratio=0.1, linewidth=1.5)
    ax.quiver(0,0,0, 0,length,0, color='black', arrow_length_ratio=0.1, linewidth=1.5)
    ax.quiver(0,0,0, 0,0,length, color='black', arrow_length_ratio=0.1, linewidth=1.5)
    ax.text(length, 0, 0, "PC1", color="black", fontsize=18)
    ax.text(0, length, 0, "PC2", color="black", fontsize=18)
    ax.text(0, 0, length, "PC3", color="black", fontsize=18)

# Compute global cosine similarity range from real and generated data
global_cos = [cosine_similarity(real_text_emb[j], real_img_emb[j]) for j in range(num_samples)]
base_npz_prefix = os.path.join(data_folder, f"CLIP_{model_fname}_embeddings_cfg")
cfg_scales = [1, 3, 7, 20]
for cfg in cfg_scales:
    gen_npz = f"{base_npz_prefix}_{cfg}.npz".replace("__", "_")
    if os.path.exists(gen_npz):
        gen_data = np.load(gen_npz)
        gen_img_emb = np.array(gen_data["image_embeddings"])
        global_cos.extend([cosine_similarity(real_text_emb[j], gen_img_emb[j]) for j in range(num_samples)])
global_min, global_max = min(global_cos), max(global_cos)
normalize = lambda x: (x - global_min)/(global_max - global_min) if global_max > global_min else x

def process_subplot(ax, cap_emb, img_emb, title):
    combined = np.concatenate([cap_emb, img_emb], axis=0)
    proj = PCA(n_components=3).fit_transform(combined)
    proj = proj / np.linalg.norm(proj, axis=1, keepdims=True)
    cap3d, img3d = proj[:num_samples], proj[num_samples:]
    avg_cos, l2m, rmg = compute_metrics(cap_emb, img_emb)
    # Use larger fontsize for better readability
    ax.set_title(f"{title}\nCOS Diss: {avg_cos:.3f}  L2M: {l2m:.3f}  RMG: {rmg:.3f}", fontsize=18)
    plot_unit_sphere(ax)
    for j in range(num_samples):
        cs = normalize(cosine_similarity(cap_emb[j], img_emb[j]))
        ax.plot([cap3d[j,0], img3d[j,0]], [cap3d[j,1], img3d[j,1]], [cap3d[j,2], img3d[j,2]],
                c="black", lw=0.5, alpha=0.7)
        ax.scatter(cap3d[j,0], cap3d[j,1], cap3d[j,2],
                   s=80, marker='o', edgecolor='black', linewidth=0.8,
                   color=plt.cm.Blues(cs))
        ax.scatter(img3d[j,0], img3d[j,1], img3d[j,2],
                   s=80, marker='s', edgecolor='black', linewidth=0.8,
                   color=plt.cm.Reds(cs))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.grid(False)
    set_axes_equal(ax)
    hide_axis_lines(ax)
    draw_coordinate_arrows(ax)

# --- Create figure with 2 rows ---
# First row: Real + default CFG scales (Real, CFG 1, CFG 3, CFG 7, CFG 20) → 5 columns
# Second row: Scale 7 with different sampling steps: 10, 20, 100, 200, 500 → 5 columns
ncols = 5
nrows = 2

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, subplot_kw={'projection': '3d'}, figsize=(30, 12))

# --- First Row (Row 0): Real data and default CFG scales ---
process_subplot(axes[0, 0], real_text_emb, real_img_emb, "Real")
legend_handles = [
    Line2D([0], [0], marker='o', color='w', label='Caption',
           markerfacecolor=plt.cm.Blues(normalize(global_cos[0])), markersize=10, markeredgecolor='black'),
    Line2D([0], [0], marker='s', color='w', label='Image',
           markerfacecolor=plt.cm.Reds(normalize(global_cos[0])), markersize=10, markeredgecolor='black')
]
axes[0, 0].legend(handles=legend_handles, fontsize=14)

for idx, cfg in enumerate(cfg_scales, start=1):
    gen_npz = f"{base_npz_prefix}_{cfg}.npz".replace("__", "_")
    if not os.path.exists(gen_npz):
        print(f"File not found: {gen_npz}. Skipping CFG {cfg}.")
        continue
    gen_data = np.load(gen_npz)
    gen_img_emb = np.array(gen_data["image_embeddings"])
    process_subplot(axes[0, idx], real_text_emb, gen_img_emb, f"CFG {cfg}")

# --- Second Row (Row 1): Scale 7 with different sampling steps ---
sampling_steps = [10, 20, 100, 200, 500]
for col_idx, step in enumerate(sampling_steps):
    gen_npz = os.path.join(data_folder, f"CLIP_{model_fname}_embeddings_cfg_7_steps_{step}.npz")
    if not os.path.exists(gen_npz):
        print(f"File not found: {gen_npz}. Skipping scale 7 with steps {step}.")
        axes[1, col_idx].axis('off')
        continue
    gen_data = np.load(gen_npz)
    gen_img_emb = np.array(gen_data["image_embeddings"])
    process_subplot(axes[1, col_idx], real_text_emb, gen_img_emb, f"Sampling Steps {step}")

plt.tight_layout(rect=[0,0,1,0.95])
output_plot = os.path.join(data_folder, "compact_no_box_pca3d_unit_sphere.png")
plt.savefig(output_plot, dpi=300)
plt.close()
print(f"Saved plot to {output_plot}")
