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

# Function to load embeddings for a specific generation
def load_generation_embeddings(gen_number):
    """Load image embeddings for a specific generation."""
    embedding_path = os.path.join(data_folder, f"gen_{gen_number}_embeddings.npz")
    
    if not os.path.exists(embedding_path):
        print(f"Warning: Embeddings for generation {gen_number} not found at {embedding_path}")
        # Return dummy data for demonstration if file doesn't exist
        return np.random.rand(100, 512), ["Dummy caption"] * 100
    
    data = np.load(embedding_path, allow_pickle=True)
    image_embeddings = data["image_embeddings"]
    
    # Load captions if available (for reference only)
    captions = []
    if "captions" in data:
        captions = data["captions"]
        print(f"Generation {gen_number} sample captions: {captions[0]}, {captions[1] if len(captions) > 1 else ''}")
    
    return image_embeddings, captions

# Function to load COCO caption embeddings
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

# Function to calculate cluster center and variance
def calculate_cluster_stats(embeddings_2d):
    """Calculate the center and variance of a cluster of embeddings."""
    center = np.mean(embeddings_2d, axis=0)
    # Calculate the variance of the cluster (sum of variances along each dimension)
    variance = np.var(embeddings_2d, axis=0).sum()
    # Calculate the standard deviation for the circle radius (average of stds in each dimension)
    std_distance = np.sqrt(variance)
    
    return center, variance, std_distance

# Load COCO caption embeddings - these are the same for all generations
coco_text_embeddings, coco_captions = load_coco_caption_embeddings()

# Load image embeddings for generations 0, 4, and 10
gen_numbers = [0, 4, 10]
all_image_embeddings = {}

for gen in gen_numbers:
    img_emb, _ = load_generation_embeddings(gen)
    all_image_embeddings[gen] = img_emb

# Combine all embeddings for dimensionality reduction
combined_embeddings = np.vstack([
    coco_text_embeddings,  # COCO caption embeddings
    all_image_embeddings[0],  # Image embeddings for gen 0
    all_image_embeddings[4],  # Image embeddings for gen 4
    all_image_embeddings[10]  # Image embeddings for gen 10
])

# Apply t-SNE for dimensionality reduction
print("Applying t-SNE dimensionality reduction...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
embedding_2d = tsne.fit_transform(combined_embeddings)

# Split the 2D embeddings back into their respective groups
n_captions = coco_text_embeddings.shape[0]
caption_2d = embedding_2d[:n_captions]
start_idx = n_captions

gen0_img_2d = embedding_2d[start_idx:start_idx + len(all_image_embeddings[0])]
start_idx += len(all_image_embeddings[0])

gen4_img_2d = embedding_2d[start_idx:start_idx + len(all_image_embeddings[4])]
start_idx += len(all_image_embeddings[4])

gen10_img_2d = embedding_2d[start_idx:start_idx + len(all_image_embeddings[10])]

# Calculate cluster centers and variances
caption_center, caption_variance, caption_std_dist = calculate_cluster_stats(caption_2d)
gen0_center, gen0_variance, gen0_std_dist = calculate_cluster_stats(gen0_img_2d)
gen4_center, gen4_variance, gen4_std_dist = calculate_cluster_stats(gen4_img_2d)
gen10_center, gen10_variance, gen10_std_dist = calculate_cluster_stats(gen10_img_2d)

# Print the variances for verification
print(f"Caption variance: {caption_variance:.4f}")
print(f"Gen 0 variance: {gen0_variance:.4f}")
print(f"Gen 4 variance: {gen4_variance:.4f}")
print(f"Gen 10 variance: {gen10_variance:.4f}")

# Use default matplotlib style for research paper
plt.style.use('default')

# Create a compact visualization for research paper
plt.figure(figsize=(8, 6), dpi=300)

# Define colors for different generations
caption_color = 'blue'
gen0_color = 'red'
gen4_color = 'green'
gen10_color = 'purple'

# Define markers and sizes
caption_marker = 'o'  # circle for captions
img_marker = 's'      # square for images
point_size = 50       # point size for research paper
alpha_points = 0.8    # point transparency
alpha_areas = 0.2     # area transparency
arrow_alpha = 0.8     # arrow transparency
arrow_width = 2       # arrow width

# Plot caption embeddings
plt.scatter(caption_2d[:, 0], caption_2d[:, 1], c=caption_color, marker=caption_marker, s=point_size, alpha=alpha_points, label='Captions', edgecolors='none')

# Plot image embeddings for each generation
plt.scatter(gen0_img_2d[:, 0], gen0_img_2d[:, 1], c=gen0_color, marker=img_marker, s=point_size, alpha=alpha_points, label='Gen 0', edgecolors='none')
plt.scatter(gen4_img_2d[:, 0], gen4_img_2d[:, 1], c=gen4_color, marker=img_marker, s=point_size, alpha=alpha_points, label='Gen 4', edgecolors='none')
plt.scatter(gen10_img_2d[:, 0], gen10_img_2d[:, 1], c=gen10_color, marker=img_marker, s=point_size, alpha=alpha_points, label='Gen 10', edgecolors='none')

# Draw cluster areas as filled circles with transparency
# Caption cluster area
caption_circle = plt.Circle((caption_center[0], caption_center[1]), caption_std_dist, 
                           color=caption_color, fill=True, alpha=alpha_areas)
plt.gca().add_patch(caption_circle)

# Gen 0 cluster area
gen0_circle = plt.Circle((gen0_center[0], gen0_center[1]), gen0_std_dist, 
                        color=gen0_color, fill=True, alpha=alpha_areas)
plt.gca().add_patch(gen0_circle)

# Gen 4 cluster area
gen4_circle = plt.Circle((gen4_center[0], gen4_center[1]), gen4_std_dist, 
                        color=gen4_color, fill=True, alpha=alpha_areas)
plt.gca().add_patch(gen4_circle)

# Gen 10 cluster area
gen10_circle = plt.Circle((gen10_center[0], gen10_center[1]), gen10_std_dist, 
                         color=gen10_color, fill=True, alpha=alpha_areas)
plt.gca().add_patch(gen10_circle)

# Draw arrows to show the direction of distribution shift
# From Gen 0 to Gen 4
arrow_0_to_4 = FancyArrowPatch(
    gen0_center, gen4_center, 
    connectionstyle="arc3,rad=0.1", 
    arrowstyle="simple", 
    color='black', 
    linewidth=arrow_width*1.0,  # Reduced from 1.5 to 1.0
    alpha=0.8,  # Reduced from 1.0 to 0.8
    mutation_scale=12  # Reduced from 15 to 12
)
plt.gca().add_patch(arrow_0_to_4)

# From Gen 4 to Gen 10
arrow_4_to_10 = FancyArrowPatch(
    gen4_center, gen10_center, 
    connectionstyle="arc3,rad=0.1", 
    arrowstyle="simple", 
    color='black', 
    linewidth=arrow_width*1.0,  # Reduced from 1.5 to 1.0
    alpha=0.8,  # Reduced from 1.0 to 0.8
    mutation_scale=12  # Reduced from 15 to 12
)
plt.gca().add_patch(arrow_4_to_10)

# Calculate positions for variance annotations to avoid overlap
# We'll position them at different angles around each center
angle_gen0 = 225  # degrees (lower left)
angle_gen4 = 315  # degrees (lower right)
angle_gen10 = 270  # degrees (directly below)

# Calculate offset distances based on std_dist but ensure minimum distance
offset_distance_gen0 = max(gen0_std_dist*1.5, 5)
offset_distance_gen4 = max(gen4_std_dist*1.5, 5)
offset_distance_gen10 = max(gen10_std_dist*1.5, 5)

# Convert angles to radians and calculate offset positions
gen0_offset_x = gen0_center[0] + offset_distance_gen0 * math.cos(math.radians(angle_gen0))
gen0_offset_y = gen0_center[1] + offset_distance_gen0 * math.sin(math.radians(angle_gen0))

gen4_offset_x = gen4_center[0] + offset_distance_gen4 * math.cos(math.radians(angle_gen4))
gen4_offset_y = gen4_center[1] + offset_distance_gen4 * math.sin(math.radians(angle_gen4))

gen10_offset_x = gen10_center[0] + offset_distance_gen10 * math.cos(math.radians(angle_gen10))
gen10_offset_y = gen10_center[1] + offset_distance_gen10 * math.sin(math.radians(angle_gen10))

# Add variance annotations with repositioned text boxes - no connecting arrows
plt.annotate(
    f"Var: {gen0_variance:.2f}", 
    xy=(gen0_offset_x, gen0_offset_y),  # Direct positioning without arrow connection
    fontsize=12, 
    color='black',
    ha='center', 
    va='center',
    bbox=dict(boxstyle="round,pad=0.3", fc='white', ec=gen0_color, alpha=0.9, linewidth=1.5)
)

plt.annotate(
    f"Var: {gen4_variance:.2f}", 
    xy=(gen4_offset_x, gen4_offset_y),  # Direct positioning without arrow connection
    fontsize=12, 
    color='black',
    ha='center', 
    va='center',
    bbox=dict(boxstyle="round,pad=0.3", fc='white', ec=gen4_color, alpha=0.9, linewidth=1.5)
)

plt.annotate(
    f"Var: {gen10_variance:.2f}", 
    xy=(gen10_offset_x, gen10_offset_y),  # Direct positioning without arrow connection
    fontsize=12, 
    color='black',
    ha='center', 
    va='center',
    bbox=dict(boxstyle="round,pad=0.3", fc='white', ec=gen10_color, alpha=0.9, linewidth=1.5)
)

# Add title and labels with larger font sizes for research paper
plt.title('Evolution of Embeddings Across Generations', fontsize=16, fontweight='bold')
plt.xlabel('t-SNE Dimension 1', fontsize=14)
plt.ylabel('t-SNE Dimension 2', fontsize=14)

# Add legend with larger font
plt.legend(fontsize=12, loc='best', framealpha=0.9)

# Remove ticks for cleaner look
plt.xticks([])
plt.yticks([])

# Ensure the aspect ratio is equal so circles appear as circles
plt.axis('equal')

# Improve layout for research paper
plt.tight_layout()

# Save the figure with higher DPI for research paper
output_path = os.path.join(output_folder, 'generation_clusters_tsne_paper.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Visualization saved to {output_path}")

# Create a bar chart to visualize the metrics - more compact for research paper
plt.figure(figsize=(8, 4), dpi=300)

# Calculate metrics between generations
metrics = {}
metrics['0_to_4'] = np.sqrt(np.sum((gen0_center - gen4_center) ** 2)), gen4_variance - gen0_variance
metrics['4_to_10'] = np.sqrt(np.sum((gen4_center - gen10_center) ** 2)), gen10_variance - gen4_variance
metrics['0_to_10'] = np.sqrt(np.sum((gen0_center - gen10_center) ** 2)), gen10_variance - gen0_variance

# Plot center distances
plt.subplot(1, 2, 1)
gen_pairs = ['Gen 0→4', 'Gen 4→10', 'Gen 0→10']
center_distances = [metrics['0_to_4'][0], metrics['4_to_10'][0], metrics['0_to_10'][0]]
colors = ['#777777', '#777777', '#555555']
plt.bar(gen_pairs, center_distances, color=colors)
plt.title('Cluster Center Movement', fontsize=14, fontweight='bold')
plt.ylabel('Distance', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Plot variance changes
plt.subplot(1, 2, 2)
var_changes = [metrics['0_to_4'][1], metrics['4_to_10'][1], metrics['0_to_10'][1]]
colors = ['#777777' if vc >= 0 else '#AA5555' for vc in var_changes]
plt.bar(gen_pairs, var_changes, color=colors)
plt.title('Variance Change', fontsize=14, fontweight='bold')
plt.ylabel('Change in Variance', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Add a horizontal line at y=0 for variance changes
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

plt.tight_layout()

# Save the metrics visualization with higher DPI for research paper
output_path_metrics = os.path.join(output_folder, 'generation_metrics_tsne_paper.png')
plt.savefig(output_path_metrics, dpi=300, bbox_inches='tight')
print(f"Metrics visualization saved to {output_path_metrics}")

print("All visualizations completed!") 