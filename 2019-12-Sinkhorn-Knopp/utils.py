import numpy as np
import matplotlib.pyplot as plt


def sample_random_scores_matrix(latent_size: int = 256):
    """
    Simulation of the Sinkhorn problem from: https://arxiv.org/pdf/1911.11763.pdf
    Returns score matrix which can be combined with trainable
    variables:

        Sij = Sij_aug + z * dustbin_mask + neg_score * neg_scores_mask

    where z and neg_score are trainable float scalars

    """
    # Simulate different number of points in the first and second image
    N = np.random.randint(11, 15)
    M = np.random.randint(5, 10)

    # Eq. (9)
    a_vec = np.array([1.0] * N + [M])
    b_vec = np.array([1.0] * M + [N])

    # Simulation for Eq. (6)
    fa_vector = np.random.randn(N, latent_size)
    fa_vector /= np.linalg.norm(fa_vector, axis=-1, keepdims=True)

    # fb vector is shuffled copy of fa with only fraction of the vectors
    # sampled
    indices = np.arange(N)
    np.random.shuffle(indices)
    pos_x_indices = indices[:M]
    neg_x_indices = indices[M:]
    fb_vector = fa_vector[pos_x_indices]

    # Eq (7)
    Sij = fa_vector @ fb_vector.T
    # Eq (8) - the last row and column will be later filled with
    # dustbin variable
    Sij_aug = np.zeros([N + 1, M + 1])
    Sij_aug[:N, :M] = Sij

    # This rows will be later filled with simulated negative score trainable
    # variable
    Sij_aug[neg_x_indices] = 0.0

    dustbin_mask = np.ones([N + 1, M + 1])
    dustbin_mask[:N, :M] = 0.0

    neg_scores_mask = np.zeros_like(dustbin_mask)
    neg_scores_mask[neg_x_indices] = 1.0
    neg_scores_mask *= 1 - dustbin_mask

    pos_mask = np.zeros_like(Sij_aug)
    neg_mask = np.zeros_like(Sij_aug)

    for yi, xi in enumerate(pos_x_indices):
        pos_mask[xi, yi] = 1.0

    for xi in neg_x_indices:
        neg_mask[xi, -1] = 1.0

    return Sij_aug, a_vec, b_vec, pos_mask, neg_mask, dustbin_mask, neg_scores_mask


def plot_matrices(matrices, figsize=None, cmap: str = "gray"):

    n = len(matrices)
    if figsize is None:
        figsize = (4 * n, 5)
    plt.figure(figsize=figsize)
    for i, matrix in enumerate(matrices):
        plt.subplot(1, n, i + 1)
        plt.imshow(matrix, cmap=cmap)
        plt.colorbar()
