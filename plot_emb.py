import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap

# -------------------------------
# 1. Load the Saved Embeddings
# -------------------------------
embeddings_file = "openbmb/MiniCPM-Llama3-V-2_5-int4_coco_embeddings.npz"
data = np.load(embeddings_file)
paired_lang_embeddings = data["lang"]  # shape: (100, hidden_dim)
paired_img_embeddings = data["img"]      # shape: (100, hidden_dim)
num_pairs = paired_lang_embeddings.shape[0]

print("Loaded language embeddings shape:", paired_lang_embeddings.shape)
print("Loaded image embeddings shape:", paired_img_embeddings.shape)

# -------------------------------
# 2. Concatenate and Reduce Dimensions with UMAP
# -------------------------------
all_embeddings = np.concatenate([paired_lang_embeddings, paired_img_embeddings], axis=0)
umap_model = umap.UMAP(n_components=2, random_state=42)
umap_embeddings = umap_model.fit_transform(all_embeddings)

# Split the UMAP results back into language and image embeddings.
lang_umap = umap_embeddings[:num_pairs]
img_umap = umap_embeddings[num_pairs:]

# -------------------------------
# 3. Plot and Save the Figure (do not show)
# -------------------------------
plt.figure(figsize=(10, 10))
plt.scatter(lang_umap[:, 0], lang_umap[:, 1], c='red', label='Language Embedding')
plt.scatter(img_umap[:, 0], img_umap[:, 1], c='blue', label='Image Embedding')

# Draw a line connecting each paired language and image embedding.
for i in range(num_pairs):
    plt.plot([lang_umap[i, 0], img_umap[i, 0]],
             [lang_umap[i, 1], img_umap[i, 1]], c='gray', lw=0.5)

plt.legend()
plt.title("UMAP of 100 Paired Language and Image Embeddings")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.savefig("embedding_plot.png")
plt.close()
print("Saved plot to embedding_plot.png")
