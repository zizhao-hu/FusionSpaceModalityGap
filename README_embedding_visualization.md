# Embedding Visualization for Generation Clusters

This set of scripts allows you to visualize how image embeddings evolve across different generations of the model training process, using captions from the COCO dataset. The visualization shows how the distribution of image embeddings changes as the model is trained over multiple generations.

## Scripts Overview

1. **extract_generation_embeddings.py**: Extracts CLIP embeddings from images and their corresponding captions from the COCO dataset.
2. **plot_generation_clusters.py**: Creates t-SNE visualizations of the embedding clusters and their evolution.

## Requirements

```
numpy
matplotlib
scikit-learn
torch
transformers
pillow
tqdm
```

Install the required packages:

```bash
pip install numpy matplotlib scikit-learn torch transformers pillow tqdm
```

## Usage

### Step 1: Extract Embeddings

First, extract the embeddings from the images and their corresponding captions:

```bash
python extract_generation_embeddings.py
```

This will:
- Load the first 100 captions from the COCO dataset (one caption per image)
- Extract CLIP embeddings for each caption
- Extract CLIP embeddings for images from generations 0, 4, and 10 using their corresponding captions
  - Uses images from the `data/coco/sd_to_sd_cfg_7_steps_50_gen_X` folders
  - Each image is paired with its corresponding caption by index
- Save the embeddings to NPZ files in the `data/embeddings` directory

### Step 2: Visualize Embedding Clusters

After extracting the embeddings, create the visualizations:

```bash
python plot_generation_clusters.py
```

This will generate two visualization files in the `data/visualizations` directory:
1. `generation_clusters_tsne_clean.png`: Shows the evolution of image embeddings across generations
2. `generation_metrics_tsne_clean.png`: Bar charts showing cluster movement and variance changes between generations

## Visualization Interpretation

The main visualization shows:

- **Captions** (blue circles): These represent the caption embeddings from the COCO dataset.
- **Image embeddings** (colored squares): These represent the image embeddings for each generation:
  - Gen 0: Red squares
  - Gen 4: Green squares
  - Gen 10: Purple squares
- **Cluster distributions** (colored transparent circles): These represent the distribution of each cluster:
  - Caption Distribution: Blue transparent circle
  - Gen 0 Distribution: Red transparent circle
  - Gen 4 Distribution: Green transparent circle
  - Gen 10 Distribution: Purple transparent circle
- **Directional arrows**: Black arrows showing the direction of distribution shift from one generation to the next
- **Variance annotations**: Text boxes showing the variance value for each generation's cluster

As training progresses through generations, you should observe:

1. **Distribution shift**: The arrows show how the cluster centers move across the embedding space
2. **Variance changes**: The annotations show how the spread of embeddings changes across generations
3. **Cluster distribution movement**: The distributions of later generation clusters may move in a consistent direction
4. **Cluster distribution size changes**: The size of the cluster distributions may change, indicating changes in cluster cohesion

## Metrics

The metrics visualization shows:

1. **Cluster Center Movement**: The distance between cluster centers of different generations
2. **Variance Change**: How the variance of image embeddings changes between generations

These metrics help quantify the evolution of image embeddings across generations.

## About t-SNE

t-SNE (t-Distributed Stochastic Neighbor Embedding) is a dimensionality reduction technique particularly well-suited for visualizing high-dimensional data. Unlike linear techniques like PCA, t-SNE can capture non-linear relationships in the data, making it ideal for visualizing complex embedding spaces.

The visualization preserves local relationships between points, meaning that points that are close in the high-dimensional space will tend to be close in the 2D visualization. This helps us see how image embeddings evolve across different generations of the model. 