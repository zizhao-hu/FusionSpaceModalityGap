import numpy as np

def rmg_cosine_dissimilarity(x, y):
    """
    Compute the RMG value from the given formula, but using
    cosine dissimilarity instead of Euclidean distance.
    
    RMG =  [ (1/N) * sum_i d(x_i, y_i) ]  /
           [ (1/(2*N*(N-1))) * ( sum_{i!=j} d(x_i, x_j) + sum_{i!=j} d(y_i, y_j) )
             + (1/N) * sum_i d(x_i, y_i ) ]

    where d(a, b) = 1 - (a·b)/(||a|| * ||b||).

    Parameters
    ----------
    x : np.ndarray of shape (N, D)
    y : np.ndarray of shape (N, D)

    Returns
    -------
    float
        The RMG value (cosine-based).
    """
    # Number of samples
    N = x.shape[0]
    
    # --- Helper to compute the row-by-row cosine dissimilarity (x_i, y_i)
    def cosine_dissim_rowwise(A, B):
        # A and B are shape (N, D).
        # We'll compute a 1D array of dissimilarities: dissim[i] = 1 - cos_sim(A[i], B[i]).
        # cos_sim(A[i], B[i]) = A[i] dot B[i] / (||A[i]|| * ||B[i]||)
        numerator = np.einsum('ij,ij->i', A, B)  # dot product row-by-row
        normA = np.linalg.norm(A, axis=1)
        normB = np.linalg.norm(B, axis=1)
        cos_sim = numerator / (normA * normB)
        return 1.0 - cos_sim  # dissimilarity
    
    # 1) Numerator = (1/N) * sum_i d(x_i, y_i)
    row_dissim_xy = cosine_dissim_rowwise(x, y)
    numerator = np.mean(row_dissim_xy)
    
    # 2) Denominator Part 1: (1/(2*N*(N-1))) * [sum_{i!=j} d(x_i, x_j) + sum_{i!=j} d(y_i, y_j)]
    #
    # We'll define a function that computes the sum of pairwise
    # cosine dissimilarities for all i, j (including i=j, though that diagonal is zero anyway).
    
    def sum_pairwise_cos_dissim(M):
        # M is shape (N, D)
        # We want sum_{i,j} [1 - cos_sim(M[i], M[j])]
        # We'll do a pairwise dot, then convert to cos_sim, then to dissim.
        
        # pairwise dot
        dot_mat = M @ M.T  # shape (N, N)
        # norms
        norms = np.linalg.norm(M, axis=1)  # shape (N,)
        norm_mat = np.outer(norms, norms)  # shape (N, N)
        # pairwise cos similarity
        cos_sim_mat = dot_mat / norm_mat
        # pairwise cos dissimilarity
        cos_dissim_mat = 1.0 - cos_sim_mat
        
        # sum over all i,j
        return np.sum(cos_dissim_mat)
    
    sum_dxx = sum_pairwise_cos_dissim(x)
    sum_dyy = sum_pairwise_cos_dissim(y)
    
    # average pairwise dissimilarity
    denom_part1 = (1.0 / (2.0 * N * (N - 1))) * (sum_dxx + sum_dyy)
    
    # 3) Denominator Part 2: (1/N) * sum_i d(x_i, y_i) [the same as the numerator]
    denom_part2 = numerator
    
    # 4) Denominator = denom_part1 + denom_part2
    denominator = denom_part1 + denom_part2
    
    # 5) Final RMG
    rmg_value = numerator / denominator
    return rmg_value
