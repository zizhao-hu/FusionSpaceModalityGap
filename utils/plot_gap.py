import os, sys, numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D
from PIL import Image

# Import color constants
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from utils.colors import *

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

# Calculate embedding variance
def calculate_variance(embeddings):
    # Calculate variance along each dimension and take the mean
    return np.mean(np.var(embeddings, axis=0))

# Setup paths/parameters – updated to use data/embeddings
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
data_folder = os.path.join("data", "embeddings")
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
    img_var = calculate_variance(img)  # Calculate image embedding variance
    return cos_dissim, l2m, rmg, img_var

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
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color=GREY_50, alpha=0.15,
                     edgecolor=GREY_400, linewidth=0.5, antialiased=True)

def draw_coordinate_arrows(ax, length=1.5):
    ax.quiver(0,0,0, length,0,0, color=GREY_800, arrow_length_ratio=0.1, linewidth=1.5)
    ax.quiver(0,0,0, 0,length,0, color=GREY_800, arrow_length_ratio=0.1, linewidth=1.5)
    ax.quiver(0,0,0, 0,0,length, color=GREY_800, arrow_length_ratio=0.1, linewidth=1.5)
    ax.text(length, 0, 0, "PC1", color=GREY_900, fontsize=20, fontweight='bold')
    ax.text(0, length, 0, "PC2", color=GREY_900, fontsize=20, fontweight='bold')
    ax.text(0, 0, length, "PC3", color=GREY_900, fontsize=20, fontweight='bold')

# Compute global cosine similarity range from real and generated data
global_cos = [cosine_similarity(real_text_emb[j], real_img_emb[j]) for j in range(num_samples)]
base_npz_prefix = os.path.join(data_folder, f"CLIP_{model_fname}_embeddings_cfg")

# Collect metrics for all CFG values and steps for the metrics plot
all_cfg_metrics = {}
all_steps_metrics = {}

# Expanded CFG scales for metrics plot
all_cfg_scales = [1, 3, 5, 7, 10, 20]
for cfg in all_cfg_scales:
    gen_npz = f"{base_npz_prefix}_{cfg}.npz".replace("__", "_")
    if os.path.exists(gen_npz):
        gen_data = np.load(gen_npz)
        gen_img_emb = np.array(gen_data["image_embeddings"])
        avg_cos, l2m, rmg, _ = compute_metrics(real_text_emb, gen_img_emb)
        img_pc_var = np.var(gen_img_emb, axis=0)
        total_pc_var = np.sum(img_pc_var)
        all_cfg_metrics[cfg] = {
            'COSD': avg_cos,
            'L2M': l2m,
            'RMG': rmg,
            'VAR': total_pc_var
        }
        # Only add to global_cos for the original CFG values used in visualization
        if cfg in [7, 20]:
            global_cos.extend([cosine_similarity(real_text_emb[j], gen_img_emb[j]) for j in range(num_samples)])

# Expanded steps values for metrics plot
all_steps = [10, 20, 100, 200, 500]
for step in all_steps:
    gen_npz = os.path.join(data_folder, f"CLIP_{model_fname}_embeddings_cfg_7_steps_{step}.npz")
    if os.path.exists(gen_npz):
        gen_data = np.load(gen_npz)
        gen_img_emb = np.array(gen_data["image_embeddings"])
        avg_cos, l2m, rmg, _ = compute_metrics(real_text_emb, gen_img_emb)
        img_pc_var = np.var(gen_img_emb, axis=0)
        total_pc_var = np.sum(img_pc_var)
        all_steps_metrics[step] = {
            'COSD': avg_cos,
            'L2M': l2m,
            'RMG': rmg,
            'VAR': total_pc_var
        }
        # Only add to global_cos for the original steps used in visualization
        if step in [10, 100, 500]:
            global_cos.extend([cosine_similarity(real_text_emb[j], gen_img_emb[j]) for j in range(num_samples)])

global_min, global_max = min(global_cos), max(global_cos)
normalize = lambda x: (x - global_min)/(global_max - global_min) if global_max > global_min else x

def process_subplot(ax, cap_emb, img_emb, title):
    combined = np.concatenate([cap_emb, img_emb], axis=0)
    pca = PCA(n_components=3)
    proj = pca.fit_transform(combined)
    proj = proj / np.linalg.norm(proj, axis=1, keepdims=True)
    cap3d, img3d = proj[:num_samples], proj[num_samples:]
    
    # Calculate variance using all three principal components of image embeddings
    img_pc_var = np.var(img3d, axis=0)
    pc1_var = img_pc_var[0]
    pc2_var = img_pc_var[1]
    pc3_var = img_pc_var[2]
    total_pc_var = np.sum(img_pc_var)
    
    avg_cos, l2m, rmg, _ = compute_metrics(cap_emb, img_emb)
    # Use even larger fontsize and bold text for better readability
    ax.set_title(f"{title}\nCOSD: {avg_cos:.3f}  L2M: {l2m:.3f}\nRMG: {rmg:.3f}  VAR: {total_pc_var:.3f}", 
                fontsize=24, fontweight='bold')
    plot_unit_sphere(ax)
    for j in range(num_samples):
        cs = normalize(cosine_similarity(cap_emb[j], img_emb[j]))
        ax.plot([cap3d[j,0], img3d[j,0]], [cap3d[j,1], img3d[j,1]], [cap3d[j,2], img3d[j,2]],
                c="black", lw=0.7, alpha=0.7)
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

# Function to create and overlay the metrics figure
def create_metrics_figure(main_output_path):
    # Create a figure with GridSpec for better control over subplot alignment
    metrics_fig = plt.figure(figsize=(10, 12))
    
    # Use GridSpec with consistent spacing parameters
    gs_metrics = gridspec.GridSpec(1, 2, figure=metrics_fig, wspace=0.15)
    
    # Left subplot for CFG values
    ax_cfg = metrics_fig.add_subplot(gs_metrics[0, 0])
    ax_cfg.set_facecolor('white')
    ax_cfg.grid(True, linestyle='--', alpha=0.3, color=GREY_300, linewidth=1.0)
    
    # Right subplot for Steps values
    ax_steps = metrics_fig.add_subplot(gs_metrics[0, 1])
    ax_steps.set_facecolor('white')
    ax_steps.grid(True, linestyle='--', alpha=0.3, color=GREY_300, linewidth=1.0)
    
    # Get real image metrics for reference line
    real_avg_cos, real_l2m, real_rmg, _ = compute_metrics(real_text_emb, real_img_emb)
    real_img_pc_var = np.var(real_img_emb, axis=0)
    real_total_pc_var = np.sum(real_img_pc_var)
    
    # Initialize legend handles and labels
    legend_handles = []
    legend_labels = []
    
    # Plot CFG vs metrics
    cfg_values = sorted(all_cfg_metrics.keys())
    if cfg_values:
        cosd_values = [all_cfg_metrics[cfg]['COSD'] for cfg in cfg_values]
        l2m_values = [all_cfg_metrics[cfg]['L2M'] for cfg in cfg_values]
        rmg_values = [all_cfg_metrics[cfg]['RMG'] for cfg in cfg_values]
        var_values = [all_cfg_metrics[cfg]['VAR'] for cfg in cfg_values]
        
        # Find best values (min for COSD, L2M, RMG; max for VAR)
        best_cosd_idx = np.argmin(cosd_values)
        best_l2m_idx = np.argmin(l2m_values)
        best_rmg_idx = np.argmin(rmg_values)
        best_var_idx = np.argmax(var_values)
        
        # Use thicker lines and larger markers - no normalization for VAR
        line1, = ax_cfg.plot(cfg_values, cosd_values, 'o-', color=PRIMARY, 
                          linewidth=3, markersize=10)
        line2, = ax_cfg.plot(cfg_values, l2m_values, 's-', color=ERROR, 
                          linewidth=3, markersize=10)
        line3, = ax_cfg.plot(cfg_values, rmg_values, '^-', color=SUCCESS, 
                          linewidth=3, markersize=10)
        line4, = ax_cfg.plot(cfg_values, var_values, 'D-', color=ACCENT, 
                          linewidth=3, markersize=10)
        
        # Mark best values with larger markers and black edge
        ax_cfg.plot(cfg_values[best_cosd_idx], cosd_values[best_cosd_idx], 'o', color=PRIMARY,
                 markersize=16, markeredgecolor=GREY_800, markeredgewidth=2.0)
        ax_cfg.plot(cfg_values[best_l2m_idx], l2m_values[best_l2m_idx], 's', color=ERROR,
                 markersize=16, markeredgecolor=GREY_800, markeredgewidth=2.0)
        ax_cfg.plot(cfg_values[best_rmg_idx], rmg_values[best_rmg_idx], '^', color=SUCCESS,
                 markersize=16, markeredgecolor=GREY_800, markeredgewidth=2.0)
        ax_cfg.plot(cfg_values[best_var_idx], var_values[best_var_idx], 'D', color=ACCENT,
                 markersize=16, markeredgecolor=GREY_800, markeredgewidth=2.0)
        
        # Add value labels above the best points - moved higher and non-bold
        ax_cfg.annotate(f"{cosd_values[best_cosd_idx]:.3f}", 
                     (cfg_values[best_cosd_idx], cosd_values[best_cosd_idx]),
                     xytext=(0, 20), textcoords='offset points', ha='center', 
                     fontsize=12, fontweight='normal', color=PRIMARY)
        
        ax_cfg.annotate(f"{l2m_values[best_l2m_idx]:.3f}", 
                     (cfg_values[best_l2m_idx], l2m_values[best_l2m_idx]),
                     xytext=(0, 20), textcoords='offset points', ha='center', 
                     fontsize=12, fontweight='normal', color=ERROR)
        
        ax_cfg.annotate(f"{rmg_values[best_rmg_idx]:.3f}", 
                     (cfg_values[best_rmg_idx], rmg_values[best_rmg_idx]),
                     xytext=(0, 20), textcoords='offset points', ha='center', 
                     fontsize=12, fontweight='normal', color=SUCCESS)
        
        ax_cfg.annotate(f"{var_values[best_var_idx]:.3f}", 
                     (cfg_values[best_var_idx], var_values[best_var_idx]),
                     xytext=(0, 20), textcoords='offset points', ha='center', 
                     fontsize=12, fontweight='normal', color=ACCENT)
        
        # Add reference lines for real image values with labels - thicker lines
        ax_cfg.axhline(y=real_avg_cos, color=PRIMARY, linestyle='--', alpha=0.7, linewidth=2)
        ax_cfg.axhline(y=real_l2m, color=ERROR, linestyle='--', alpha=0.7, linewidth=2)
        ax_cfg.axhline(y=real_rmg, color=SUCCESS, linestyle='--', alpha=0.7, linewidth=2)
        ax_cfg.axhline(y=real_total_pc_var, color=ACCENT, linestyle='--', alpha=0.7, linewidth=2)
  
        # Collect legend handles
        legend_handles = [line1, line2, line3, line4]
        legend_labels = ['COSD', 'L2M', 'RMG', 'VAR']
        
        # Store y-axis limits to synchronize with steps plot
        y_min, y_max = ax_cfg.get_ylim()
    
    # Plot Steps vs metrics
    steps_values = sorted(all_steps_metrics.keys())
    if steps_values:
        cosd_values = [all_steps_metrics[step]['COSD'] for step in steps_values]
        l2m_values = [all_steps_metrics[step]['L2M'] for step in steps_values]
        rmg_values = [all_steps_metrics[step]['RMG'] for step in steps_values]
        var_values = [all_steps_metrics[step]['VAR'] for step in steps_values]
        
        # Find best values (min for COSD, L2M, RMG; max for VAR)
        best_cosd_idx = np.argmin(cosd_values)
        best_l2m_idx = np.argmin(l2m_values)
        best_rmg_idx = np.argmin(rmg_values)
        best_var_idx = np.argmax(var_values)
        
        # Use thicker lines and larger markers - no normalization for VAR
        ax_steps.plot(steps_values, cosd_values, 'o-', color=PRIMARY, 
                    linewidth=3, markersize=10)
        ax_steps.plot(steps_values, l2m_values, 's-', color=ERROR, 
                    linewidth=3, markersize=10)
        ax_steps.plot(steps_values, rmg_values, '^-', color=SUCCESS, 
                    linewidth=3, markersize=10)
        ax_steps.plot(steps_values, var_values, 'D-', color=ACCENT, 
                    linewidth=3, markersize=10)
        
        # Mark best values with larger markers and black edge
        ax_steps.plot(steps_values[best_cosd_idx], cosd_values[best_cosd_idx], 'o', color=PRIMARY,
                   markersize=16, markeredgecolor=GREY_800, markeredgewidth=2.0)
        ax_steps.plot(steps_values[best_l2m_idx], l2m_values[best_l2m_idx], 's', color=ERROR,
                   markersize=16, markeredgecolor=GREY_800, markeredgewidth=2.0)
        ax_steps.plot(steps_values[best_rmg_idx], rmg_values[best_rmg_idx], '^', color=SUCCESS,
                   markersize=16, markeredgecolor=GREY_800, markeredgewidth=2.0)
        ax_steps.plot(steps_values[best_var_idx], var_values[best_var_idx], 'D', color=ACCENT,
                   markersize=16, markeredgecolor=GREY_800, markeredgewidth=2.0)
        
        # Add value labels above the best points - moved higher and non-bold
        ax_steps.annotate(f"{cosd_values[best_cosd_idx]:.3f}", 
                       (steps_values[best_cosd_idx], cosd_values[best_cosd_idx]),
                       xytext=(0, 20), textcoords='offset points', ha='center', 
                       fontsize=12, fontweight='normal', color=PRIMARY)
        
        ax_steps.annotate(f"{l2m_values[best_l2m_idx]:.3f}", 
                       (steps_values[best_l2m_idx], l2m_values[best_l2m_idx]),
                       xytext=(0, 20), textcoords='offset points', ha='center', 
                       fontsize=12, fontweight='normal', color=ERROR)
        
        ax_steps.annotate(f"{rmg_values[best_rmg_idx]:.3f}", 
                       (steps_values[best_rmg_idx], rmg_values[best_rmg_idx]),
                       xytext=(0, 20), textcoords='offset points', ha='center', 
                       fontsize=12, fontweight='normal', color=SUCCESS)
        
        ax_steps.annotate(f"{var_values[best_var_idx]:.3f}", 
                       (steps_values[best_var_idx], var_values[best_var_idx]),
                       xytext=(0, 20), textcoords='offset points', ha='center', 
                       fontsize=12, fontweight='normal', color=ACCENT)
        
        # Add reference lines for real image values with increased visibility - thicker lines
        ax_steps.axhline(y=real_avg_cos, color=PRIMARY, linestyle='--', alpha=0.7, linewidth=2)
        ax_steps.axhline(y=real_l2m, color=ERROR, linestyle='--', alpha=0.7, linewidth=2)
        ax_steps.axhline(y=real_rmg, color=SUCCESS, linestyle='--', alpha=0.7, linewidth=2)
        ax_steps.axhline(y=real_total_pc_var, color=ACCENT, linestyle='--', alpha=0.7, linewidth=2)
        
        # Add text labels for real values on the right side - larger text
        ax_steps.text(steps_values[-1], real_avg_cos, f"Real: {real_avg_cos:.3f}", color=PRIMARY, fontsize=12, 
                   fontweight='bold', ha='right', va='bottom')
        ax_steps.text(steps_values[-1], real_l2m, f"Real: {real_l2m:.3f}", color=ERROR, fontsize=12, 
                   fontweight='bold', ha='right', va='bottom')
        # Position RMG label below the dash line
        ax_steps.text(steps_values[-1], real_rmg, f"Real: {real_rmg:.3f}", color=SUCCESS, fontsize=12, 
                   fontweight='bold', ha='right', va='top')
        ax_steps.text(steps_values[-1], real_total_pc_var, f"Real: {real_total_pc_var:.3f}", color=ACCENT, fontsize=12, 
                   fontweight='bold', ha='right', va='bottom')
    
    # Set labels with larger font sizes - only x-axis labels, no titles or y-axis labels
    ax_cfg.set_xlabel('CFG', fontsize=26, fontweight='bold')
    ax_steps.set_xlabel('Steps', fontsize=26, fontweight='bold')
    
    # Synchronize y-axis limits between the two plots
    if 'y_min' in locals() and 'y_max' in locals():
        # Add a bit more padding to accommodate the value labels
        padding = (y_max - y_min) * 0.15
        ax_cfg.set_ylim(y_min, y_max + padding)
        ax_steps.set_ylim(y_min, y_max + padding)
    
    # Increase tick label size and make bold
    for label in ax_cfg.get_xticklabels() + ax_cfg.get_yticklabels():
        label.set_fontweight('bold')
    for label in ax_steps.get_xticklabels():
        label.set_fontweight('bold')
    
    ax_cfg.tick_params(axis='both', which='major', labelsize=24, width=2.5)
    ax_steps.tick_params(axis='x', which='major', labelsize=24, width=2.5)  # Only show x-axis ticks
    ax_steps.set_yticklabels([])  # Hide y-axis tick labels
    
    # Add a single legend outside the plots - position it higher and make bold
    if legend_handles:
        legend = metrics_fig.legend(legend_handles, legend_labels, 
                         loc='upper center', bbox_to_anchor=(0.5, 0.97),
                         ncol=4, fontsize=24, frameon=True)
    
    # Use constrained_layout instead of tight_layout for better alignment
    metrics_fig.set_constrained_layout(True)
    
    # Save the metrics figure to a temporary file
    metrics_temp_file = "temp_metrics_plot.png"
    metrics_fig.savefig(metrics_temp_file, dpi=300)
    plt.close(metrics_fig)
    
    # Now combine the two images
    # Open the main figure
    main_img = Image.open(main_output_path)
    
    # Open the metrics figure
    metrics_img = Image.open(metrics_temp_file)
    
    # Resize metrics image if needed - maintain original size ratio
    metrics_width = int(main_img.width * 0.35)  # 35% of main image width
    metrics_height = int(metrics_width * (metrics_img.height / metrics_img.width) * 0.9)  # Back to 30% taller
    metrics_img = metrics_img.resize((metrics_width, metrics_height), Image.LANCZOS)
    
    # Calculate position to place metrics image (top-left corner) - move it more to the left
    x_position = 0  # 2% from left edge (reduced from 5%)
    y_position = int(main_img.height * 0.45)  # 45% from top edge
    
    # Create a new image with the same size as the main image
    combined_img = main_img.copy()
    
    # Paste the metrics image onto the main image
    combined_img.paste(metrics_img, (x_position, y_position))
    
    # Save the combined image
    combined_img.save(main_output_path)
    
    # Clean up temporary files
    import os
    if os.path.exists(metrics_temp_file):
        os.remove(metrics_temp_file)

# --- Create figure with 2 rows ---
# First row: Real, CFG 7, CFG 20
# Second row: CFG 7 with steps 10 and 500 (metrics will be overlaid separately)
ncols = 3
nrows = 2

# Create figure with GridSpec for better control over subplot alignment
fig = plt.figure(figsize=(18, 12))
# Use consistent spacing parameters for perfect alignment
gs = plt.GridSpec(nrows, ncols, figure=fig, wspace=0.0, hspace=0.0)

# Create 3D axes for all plots with consistent parameters
axes = {}
for row in range(nrows):
    for col in range(ncols):
        # Create 3D subplot for all positions
        axes[(row, col)] = fig.add_subplot(gs[row, col], projection='3d')
        # Set consistent view angle for all 3D plots
        axes[(row, col)].view_init(elev=30, azim=45)

# Hide the 3D subplot in the second row first column
axes[(1, 0)].axis('off')

# --- First Row: Real, CFG 7, and CFG 20 ---
process_subplot(axes[(0, 0)], real_text_emb, real_img_emb, "Real")
legend_handles = [
    Line2D([0], [0], marker='o', color='w', label='Caption',
           markerfacecolor=plt.cm.Blues(normalize(global_cos[0])), markersize=12, markeredgecolor='black'),
    Line2D([0], [0], marker='s', color='w', label='Image',
           markerfacecolor=plt.cm.Reds(normalize(global_cos[0])), markersize=12, markeredgecolor='black')
]
# Move legend to a better position to save space
axes[(0, 0)].legend(handles=legend_handles, fontsize=24, frameon=True, loc='upper left')

# Ensure all 3D plots have the same axis limits for perfect alignment
def set_common_3d_limits(axes_dict):
    """Set the same axis limits for all 3D axes in the dictionary."""
    # Find the min and max for all axes
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')
    
    for key, ax in axes_dict.items():
        if key == (1, 0):  # Skip the hidden subplot
            continue
        
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        z_lim = ax.get_zlim()
        
        x_min = min(x_min, x_lim[0])
        x_max = max(x_max, x_lim[1])
        y_min = min(y_min, y_lim[0])
        y_max = max(y_max, y_lim[1])
        z_min = min(z_min, z_lim[0])
        z_max = max(z_max, z_lim[1])
    
    # Set the same limits for all axes
    for key, ax in axes_dict.items():
        if key == (1, 0):  # Skip the hidden subplot
            continue
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

# Raise the z-order of all left plots to be on top of right plots
for row in range(nrows):
    for col in range(ncols):
        if (row, col) in axes:
            # Skip the metrics plot which is 2D
            if row == 1 and col == 0:
                continue
            # Set zorder higher for left plots
            if row == 1 and col == 0:
                axes[(row, col)].set_zorder(20)  # Highest zorder for metrics plot
            elif col == 0:
                axes[(row, col)].set_zorder(10)  # Higher zorder means on top
            elif col == 1:
                axes[(row, col)].set_zorder(5)
            else:
                axes[(row, col)].set_zorder(1)

# Process CFG 7
cfg = 7
gen_npz = f"{base_npz_prefix}_{cfg}.npz".replace("__", "_")
if os.path.exists(gen_npz):
    gen_data = np.load(gen_npz)
    gen_img_emb = np.array(gen_data["image_embeddings"])
    process_subplot(axes[(0, 1)], real_text_emb, gen_img_emb, f"CFG {cfg}")
else:
    print(f"File not found: {gen_npz}. Skipping CFG {cfg}.")
    axes[(0, 1)].axis('off')

# Process CFG 20
cfg = 20
gen_npz = f"{base_npz_prefix}_{cfg}.npz".replace("__", "_")
if os.path.exists(gen_npz):
    gen_data = np.load(gen_npz)
    gen_img_emb = np.array(gen_data["image_embeddings"])
    process_subplot(axes[(0, 2)], real_text_emb, gen_img_emb, f"CFG {cfg}")
else:
    print(f"File not found: {gen_npz}. Skipping CFG {cfg}.")
    axes[(0, 2)].axis('off')

# Plot CFG 7 with steps 10 in the second column - no position adjustment
step = 10
gen_npz = os.path.join(data_folder, f"CLIP_{model_fname}_embeddings_cfg_7_steps_{step}.npz")
if os.path.exists(gen_npz):
    gen_data = np.load(gen_npz)
    gen_img_emb = np.array(gen_data["image_embeddings"])
    process_subplot(axes[(1, 1)], real_text_emb, gen_img_emb, f"CFG 7, Steps {step}")
else:
    print(f"File not found: {gen_npz}. Skipping CFG 7 with steps {step}.")
    axes[(1, 1)].axis('off')

# Plot CFG 7 with steps 500 in the third column - no position adjustment
step = 500
gen_npz = os.path.join(data_folder, f"CLIP_{model_fname}_embeddings_cfg_7_steps_{step}.npz")
if os.path.exists(gen_npz):
    gen_data = np.load(gen_npz)
    gen_img_emb = np.array(gen_data["image_embeddings"])
    process_subplot(axes[(1, 2)], real_text_emb, gen_img_emb, f"CFG 7, Steps {step}")
else:
    print(f"File not found: {gen_npz}. Skipping CFG 7 with steps {step}.")
    axes[(1, 2)].axis('off')

# Apply common limits to all 3D plots for perfect alignment
set_common_3d_limits(axes)

# Use constrained_layout instead of subplots_adjust for better alignment
fig.set_constrained_layout(True)
output_plot = os.path.join(data_folder, "real_cfg7_cfg20_steps_pca3d_unit_sphere.png")

# Save the main figure without the metrics plot
plt.savefig(output_plot, dpi=300)

# Create a separate metrics figure
create_metrics_figure(output_plot)

print(f"Saved plot to {output_plot}")
plt.close()
