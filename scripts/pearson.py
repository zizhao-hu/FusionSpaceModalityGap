import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load your saved embeddings.
# These arrays should be from your previous processing.
# final_text_embeddings: (100, 512)
# final_image_embeddings: (100, 512)
# seq_text_embeddings: (100, 77, 512)
# seq_image_embeddings: (100, 50, 512)  <-- note: image tokens have been projected to 512!
data = np.load('CLIP_openai_clip-vit-base-patch32_coco_embeddings.npz')
final_text_embeddings = data['final_text']   # shape (100, 512)
final_image_embeddings = data['final_image'] # shape (100, 512)
seq_text_embeddings = data['seq_text']         # shape (100, 77, 512)
seq_image_embeddings = data['seq_image']       # shape (100, 50, 512)

# 1. Compute the CLS (final) cosine similarity per sample.
# (Since the embeddings are L2-normalized, the dot product equals the cosine similarity.)
cls_cosine_sim = np.sum(final_text_embeddings * final_image_embeddings, axis=1)  # shape: (100,)

# 2. For each sample, compute the token-pair cosine similarity matrix and extract
#    the accumulated singular value ratio for the first 5 dimensions.
#    (For each sample, the text tokens have shape (77, 512) and image tokens (50, 512),
#     so the token similarity matrix is of shape (77, 50).)
accum_singular_ratios = []
for i in range(100):
    text_seq = seq_text_embeddings[i]    # shape: (77, 512)
    image_seq = seq_image_embeddings[i]    # shape: (50, 512)
    
    # Compute the token-level cosine similarity matrix.
    # Since tokens are normalized, the dot product is the cosine similarity.
    sim_matrix = np.dot(text_seq, image_seq.T)  # shape: (77, 50)
    
    # Compute SVD and extract the singular values.
    U, S, Vh = np.linalg.svd(sim_matrix, full_matrices=False)
    # Compute the ratio of the sum of the top 5 singular values to the sum of all singular values.
    ratio = np.sum(S[:5]) / np.sum(S)
    accum_singular_ratios.append(ratio)

accum_singular_ratios = np.array(accum_singular_ratios)  # shape: (100,)

# 3. Compute the Pearson correlation coefficient between the CLS cosine similarities
#    and the accumulated singular value ratios.
pearson_coef, p_value = pearsonr(cls_cosine_sim, accum_singular_ratios)
print("Pearson Correlation Coefficient:", pearson_coef)
print("P-value:", p_value)

# 4. Plot the scatter plot with a regression line.
plt.figure(figsize=(8, 6))
plt.scatter(cls_cosine_sim, accum_singular_ratios, label='Samples', color='blue')

# Fit a regression line.
slope, intercept = np.polyfit(cls_cosine_sim, accum_singular_ratios, 1)
regression_line = slope * cls_cosine_sim + intercept
plt.plot(cls_cosine_sim, regression_line, color='red',
         label=f'Regression line\nPearson r = {pearson_coef:.3f}')

plt.xlabel('CLS Cosine Similarity')
plt.ylabel('Accumulated Singular Value Ratio (Top 5 / All)')
plt.title('Correlation between CLS Similarity and Accumulated Sequence Similarity Ratio')
plt.legend()
plt.grid(True)
plt.show()
