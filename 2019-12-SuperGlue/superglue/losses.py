import tensorflow as tf
import numpy as np


def py_compute_optimal_transport(
    M: np.ndarray, r: np.ndarray, c: np.ndarray, lam: float, epsilon=1e-5
):
    """
    Computes the optimal transport matrix and Slinkhorn distance using the
    Sinkhorn-Knopp algorithm

    @author: Michiel Stock
    michielfmstock@gmail.com


    Inputs:
        - M : cost matrix (n x m)
        - r : vector of marginals (n, )
        - c : vector of marginals (m, )
        - lam : strength of the entropic regularization
        - epsilon : convergence parameter

    Output:
        - P : optimal transport matrix (n x m)
        - dist : Sinkhorn distance
    """
    n, m = M.shape
    P = np.exp(-lam * M)
    P /= P.sum()
    u = np.zeros(n)
    # normalize this matrix
    while np.max(np.abs(u - P.sum(1))) > epsilon:
        u = P.sum(1)
        P *= (r / u).reshape((-1, 1))
        P *= (c / P.sum(0)).reshape((1, -1))
    return P, np.sum(P * M)


def sinkhorn_step(Pij: tf.Tensor, a_vec: tf.Tensor, b_vec: tf.Tensor) -> tf.Tensor:
    batch_size = tf.shape(Pij)[0]
    a_vec = tf.cast(a_vec, dtype=Pij.dtype)
    b_vec = tf.cast(b_vec, dtype=Pij.dtype)

    row_sum = tf.reduce_sum(Pij, axis=1)  # (batch_size, num_keypoints)
    col_sum = tf.reduce_sum(Pij, axis=2)  # (batch_size, num_keypoints)

    u = tf.reshape(a_vec / col_sum, [batch_size, -1, 1])
    Pij = Pij * u
    v = tf.reshape(b_vec / row_sum, [batch_size, 1, -1])
    Pij = Pij * v
    return Pij


def normalize_pij(Pij):
    norm_const = tf.math.reduce_sum(Pij, [1, 2], keepdims=True)
    return Pij / norm_const


def batch_compute_optimal_transport(
    M: tf.Tensor, r: tf.Tensor, c: tf.Tensor, lam: float, num_steps: int
):
    """
    Computes the optimal transport matrix and Sinkhorn distance using the
    Sinkhorn-Knopp algorithm

    Inputs:
        - M : cost matrix (batch_size, n x m)
        - r : vector of marginals (1, n)
        - c : vector of marginals (1, m)
        - lam : strength of the entropic regularization
        - epsilon : convergence parameter

    Output:
        - P : optimal transport matrix (n x m)
        - dist : Sinkhorn distance
    """

    P = tf.exp(-lam * M)
    P = normalize_pij(P)

    for i in range(num_steps):
        P = sinkhorn_step(P, r, c)

    return P, tf.reduce_sum(P * M, axis=[1, 2])


class SuperGlueSinkhornKnoppLayer(tf.keras.layers.Layer):
    def __init__(self, lam: float = 5.0, num_steps: int = 100, dustbin_init = 0.0):
        super(SuperGlueSinkhornKnoppLayer, self).__init__()
        self.lam = lam
        self.num_steps = num_steps
        self.dustbin_variable = tf.Variable(dustbin_init)

    def call(self, fA, fB):
        batch_size, num_keypoints, latent_size = fA.shape.as_list()
        assert fA.shape == fB.shape

        fA = tf.math.l2_normalize(fA, axis=-1)
        fB = tf.math.l2_normalize(fB, axis=-1)

        # (batch_size, num_keypoints, num_keypoints)
        Sij = tf.matmul(fA, fB, transpose_b=True)

        row = self.dustbin_variable * tf.ones([batch_size, 1, num_keypoints])
        col = self.dustbin_variable * tf.ones([batch_size, num_keypoints + 1, 1])
        Sij_aug = tf.concat([Sij, row], axis=1)
        Sij_aug = tf.concat([Sij_aug, col], axis=2)  # [batch_size, nk +1, nk + 1]

        a_vec = np.array([[1.0] * num_keypoints + [num_keypoints]])
        b_vec = np.array([[1.0] * num_keypoints + [num_keypoints]])
        a_vec = tf.convert_to_tensor(a_vec)
        b_vec = tf.convert_to_tensor(b_vec)

        Cij = - Sij_aug # convert scores matrix to cost matrix
        Cij = tf.clip_by_value(Cij, -15 / self.lam, +15 / self.lam)
        # solve optimal transport using python solver to find good initial conditions
        # for the graph based approach.
        Pij_optimal, _ = batch_compute_optimal_transport(
            Cij, a_vec, b_vec, lam=self.lam, num_steps=self.num_steps
        )
        # estimate good initial conditions to be used in the graph approach, so we will
        # need only one iteration to reach convergence
        alpha_beta = Pij_optimal * tf.exp(self.lam * Cij)
        alpha_beta = tf.stop_gradient(alpha_beta)

        Pij_0 = alpha_beta * tf.exp( - self.lam * Cij)
        # Pij_0 = normalize_pij(Pij_0)
        # Pij = sinkhorn_step(Pij_0, a_vec, b_vec)
        Pij = sinkhorn_step(Pij_optimal, a_vec, b_vec)
        print(Pij_optimal[0] - Pij[0])
        return Pij_optimal#Pij
