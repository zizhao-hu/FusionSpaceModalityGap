import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from PIL import Image
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import math

# Set up paths
data_folder = os.path.join("data", "embeddings")
output_folder = os.path.join("data", "visualizations")
os.makedirs(output_folder, exist_ok=True)

def load_generation_embeddings(gen_number):
    """Load image embeddings for a specific generation."""
    embedding_path = os.path.join(data_folder, f"gen_{gen_number}_embeddings.npz")
    
    if not os.path.exists(embedding_path):
        print(f"Warning: Embeddings for generation {gen_number} not found at {embedding_path}")
        # Return dummy data for demonstration if file doesn't exist
        return np.random.rand(100, 512), ["Dummy caption"] * 100
    
    data = np.load(embedding_path, allow_pickle=True)
    image_embeddings = data["image_embeddings"]
    
    captions = []
    if "captions" in data:
        captions = data["captions"]
        print(f"Generation {gen_number} sample captions: {captions[0]}, {captions[1] if len(captions) > 1 else ''}")
    
    return image_embeddings, captions

def load_coco_caption_embeddings():
    """Load the COCO caption embeddings."""
    embedding_path = os.path.join(data_folder, "coco_caption_embeddings.npz")
    
    if not os.path.exists(embedding_path):
        print(f"Warning: COCO caption embeddings not found at {embedding_path}")
        # Return dummy data for demonstration if file doesn't exist
        return np.random.rand(100, 512), ["Dummy caption"] * 100
    
    data = np.load(embedding_path, allow_pickle=True)
    text_embeddings = data["text_embeddings"]
    captions = data["captions"]
    
    print(f"Loaded {len(text_embeddings)} COCO caption embeddings")
    print(f"Sample captions: {captions[0]}, {captions[1]}, {captions[2]}...")
    
    return text_embeddings, captions

def calculate_cluster_stats(embeddings_2d):
    """Calculate the center and variance of a cluster of embeddings."""
    center = np.mean(embeddings_2d, axis=0)
    variance = np.var(embeddings_2d, axis=0).sum()
    std_distance = np.sqrt(variance)
    return center, variance, std_distance

def create_shifted_embeddings(base_embeddings, shift_vector, variance_scale):
    """Create shifted embeddings with adjusted variance."""
    shifted = base_embeddings + shift_vector
    # Adjust variance by scaling distances from mean
    mean = np.mean(shifted, axis=0)
    centered = shifted - mean
    scaled = centered * np.sqrt(variance_scale)
    return scaled + mean

def plot_scenario(ax, caption_2d, gen0_img_2d, gen4_img_2d, gen10_img_2d, 
                 additional_captions_2d=None, title="", show_legend=True,
                 caption_labels=None):
    """Create a single plot for one scenario."""
    # Define colors and styles
    caption_color = 'midnightblue'  # Deep blue for base captions
    gen0_color = 'red'
    gen4_color = 'green'
    gen10_color = 'purple'
    
    # Define more drastic blue gradient colors for additional captions
    blue_gradients = ['lightskyblue', 'aliceblue', 'white']
    # Edge colors for dots (especially needed for white)
    edge_colors = ['deepskyblue', 'steelblue', 'navy']
    
    point_size = 30
    alpha_points = 0.8
    alpha_areas = 0.15
    arrow_alpha = 0.8
    arrow_width = 1.5
    
    # Plot points
    ax.scatter(caption_2d[:, 0], caption_2d[:, 1], c=caption_color, marker='o', 
              s=point_size, alpha=alpha_points, label='Captions', edgecolors='none')
    
    # Plot additional caption points if provided
    if additional_captions_2d is not None:
        for i, add_captions in enumerate(additional_captions_2d):
            label = caption_labels[i] if caption_labels else f'Tagged {i+1}'
            # Use blue gradient colors instead of default colors
            color = blue_gradients[i % len(blue_gradients)]
            edge_color = edge_colors[i % len(edge_colors)]
            ax.scatter(add_captions[:, 0], add_captions[:, 1], c=color, 
                      marker='o', s=point_size, alpha=alpha_points, 
                      label=label, edgecolors=edge_color, linewidths=1)
    
    # Plot image embeddings
    ax.scatter(gen0_img_2d[:, 0], gen0_img_2d[:, 1], c=gen0_color, marker='s', 
              s=point_size, alpha=alpha_points, label='Gen 0', edgecolors='none')
    ax.scatter(gen4_img_2d[:, 0], gen4_img_2d[:, 1], c=gen4_color, marker='s', 
              s=point_size, alpha=alpha_points, label='Gen 4', edgecolors='none')
    ax.scatter(gen10_img_2d[:, 0], gen10_img_2d[:, 1], c=gen10_color, marker='s', 
              s=point_size, alpha=alpha_points, label='Gen 10', edgecolors='none')
    
    # Calculate and plot cluster statistics
    def plot_cluster_stats(embeddings, color, label, i=None):
        center, variance, std_dist = calculate_cluster_stats(embeddings)
        # Plot cluster area
        if color == 'white' and i is not None:
            edge_color = edge_colors[i % len(edge_colors)]
            circle = plt.Circle((center[0], center[1]), std_dist, 
                              color=color, fill=True, alpha=alpha_areas,
                              edgecolor=edge_color, linewidth=1)
        else:
            circle = plt.Circle((center[0], center[1]), std_dist, 
                              color=color, fill=True, alpha=alpha_areas)
        ax.add_patch(circle)
        # Plot small circle at cluster center
        ax.plot(center[0], center[1], 'o', color=color, markersize=8, 
               markeredgecolor='black', markeredgewidth=1)
        return center, variance
    
    # Plot cluster stats for all embeddings
    caption_center, _ = plot_cluster_stats(caption_2d, caption_color, "Captions")
    gen0_center, gen0_var = plot_cluster_stats(gen0_img_2d, gen0_color, "Gen 0")
    gen4_center, gen4_var = plot_cluster_stats(gen4_img_2d, gen4_color, "Gen 4")
    gen10_center, gen10_var = plot_cluster_stats(gen10_img_2d, gen10_color, "Gen 10")
    
    # Plot additional caption cluster stats if provided
    if additional_captions_2d is not None:
        for i, add_captions in enumerate(additional_captions_2d):
            label = caption_labels[i] if caption_labels else f'Tagged {i+1}'
            # Use blue gradient colors instead of default colors
            color = blue_gradients[i % len(blue_gradients)]
            plot_cluster_stats(add_captions, color, label, i)
    
    # Add arrows between cluster centers
    arrow = FancyArrowPatch(gen0_center, gen4_center, arrowstyle='->', 
                          color='black', alpha=arrow_alpha, linewidth=arrow_width)
    ax.add_patch(arrow)
    arrow = FancyArrowPatch(gen4_center, gen10_center, arrowstyle='->', 
                          color='black', alpha=arrow_alpha, linewidth=arrow_width)
    ax.add_patch(arrow)
    
    # Remove variance annotations (as requested)
    
    # Add title and legend
    ax.set_title(title, fontsize=14)
    if show_legend:
        ax.legend(loc='upper right')
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

# Load original embeddings
coco_text_embeddings, coco_captions = load_coco_caption_embeddings()
gen_numbers = [0, 4, 10]
all_image_embeddings = {}
for gen in gen_numbers:
    img_emb, _ = load_generation_embeddings(gen)
    all_image_embeddings[gen] = img_emb

# Apply t-SNE to original embeddings
print("Applying t-SNE dimensionality reduction...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
original_2d = tsne.fit_transform(np.vstack([
    coco_text_embeddings,
    all_image_embeddings[0],
    all_image_embeddings[4],
    all_image_embeddings[10]
]))

# Split original 2D embeddings
n_captions = len(coco_text_embeddings)
caption_2d = original_2d[:n_captions]
start_idx = n_captions
gen0_img_2d = original_2d[start_idx:start_idx + len(all_image_embeddings[0])]
start_idx += len(all_image_embeddings[0])
gen4_img_2d = original_2d[start_idx:start_idx + len(all_image_embeddings[4])]
start_idx += len(all_image_embeddings[4])
gen10_img_2d = original_2d[start_idx:]

# Create synthetic data for scenarios 2 and 3
def create_scenario_data(base_caption_2d, base_gen4_2d, base_gen10_2d, 
                        caption_shifts, caption_vars, img_shift_scale, 
                        gen4_var_scale, gen10_var_scale):  # Separate scales for Gen 4 and Gen 10
    """Create synthetic data for a scenario."""
    additional_captions = []
    for shift, var_scale in zip(caption_shifts, caption_vars):
        shifted_captions = create_shifted_embeddings(
            base_caption_2d,
            shift * np.array([1, 0.5]),  # Shift vector
            var_scale  # Variance scale
        )
        additional_captions.append(shifted_captions)
    
    # Create slightly shifted image embeddings with different scales
    gen4_img = create_shifted_embeddings(
        base_gen4_2d,
        img_shift_scale * np.array([0.5, 0.25]),
        gen4_var_scale  # Scale for Gen 4
    )
    gen10_img = create_shifted_embeddings(
        base_gen10_2d,
        img_shift_scale * np.array([1, 0.5]),
        gen10_var_scale  # Scale for Gen 10
    )
    
    return additional_captions, gen4_img, gen10_img

# Create data for scenario 1 (original)
# This is the baseline - no modifications needed

# Create data for scenario 2 (single tagged caption)
scenario2_captions, scenario2_gen4, scenario2_gen10 = create_scenario_data(
    caption_2d, gen4_img_2d, gen10_img_2d,
    caption_shifts=[1.5],  # Moderate shift for caption
    caption_vars=[1.0],  # Same variance as base captions
    img_shift_scale=0.4,  # Reduced shift between generations
    gen4_var_scale=1.05,  # Gen 4 scaled by 1.05x compared to figure 1
    gen10_var_scale=1.15  # Gen 10 scaled by 1.15x compared to figure 1
)

# Create data for scenario 3 (relabeled captions)
scenario3_captions, scenario3_gen4, scenario3_gen10 = create_scenario_data(
    caption_2d, gen4_img_2d, gen10_img_2d,
    caption_shifts=[2.0, 2.5],  # Progressive shifts for gen 4 and gen 10
    caption_vars=[1.4, 1.6],  # More gradual increase in variance
    img_shift_scale=0.2,  # Minimal shift between generations
    gen4_var_scale=1.1,  # Gen 4 scaled by 1.1x compared to figure 1
    gen10_var_scale=1.6  # Gen 10 scaled by 1.6x compared to figure 1
)

# Create figure with three subplots - make them closer together with minimal gaps
plt.figure(figsize=(18, 5), dpi=300)
plt.subplots_adjust(wspace=0.01, left=0.02, right=0.98, top=0.95, bottom=0.05)  # Minimize all spacing

# Plot scenario 1 (original)
ax1 = plt.subplot(131)
# Note: gen0_img_2d is used directly in all scenarios to ensure Gen 0 is the same size across all figures
plot_scenario(ax1, caption_2d, gen0_img_2d, gen4_img_2d, gen10_img_2d,
             title="Single Recursive", show_legend=True)

# Plot scenario 2 (single tagged caption)
ax2 = plt.subplot(132)
plot_scenario(ax2, caption_2d, gen0_img_2d, scenario2_gen4, scenario2_gen10,
             additional_captions_2d=scenario2_captions,
             caption_labels=["Tagged Caption"],
             title="Recursive with Tagging", show_legend=True)

# Plot scenario 3 (relabeled captions)
ax3 = plt.subplot(133)
plot_scenario(ax3, caption_2d, gen0_img_2d, scenario3_gen4, scenario3_gen10,
             additional_captions_2d=scenario3_captions,
             caption_labels=["Gen 4", "Gen 10"],
             title="Recursive with Regrounding", show_legend=True)

# After creating all three subplots, set the same axis limits for all
def set_common_limits(axes_list):
    """Set the same axis limits for all axes in the list."""
    # Find the min and max for all axes
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    
    for ax in axes_list:
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        x_min = min(x_min, x_lim[0])
        x_max = max(x_max, x_lim[1])
        y_min = min(y_min, y_lim[0])
        y_max = max(y_max, y_lim[1])
    
    # Add a bit of padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= 0.05 * x_range  # Less padding to make figures closer
    x_max += 0.05 * x_range
    y_min -= 0.05 * y_range
    y_max += 0.05 * y_range
    
    # Set the same limits for all axes
    for ax in axes_list:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

# Set common limits for all three subplots
set_common_limits([ax1, ax2, ax3])

# Use tight_layout with minimal padding
plt.tight_layout(pad=0.5, w_pad=0.1)  # Reduce padding between subplots
output_path = os.path.join(output_folder, 'generation_clusters_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Comparison visualization saved to {output_path}")
print("Visualization completed!") 