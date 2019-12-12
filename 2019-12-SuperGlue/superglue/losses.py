from pprint import pprint

import tensorflow as tf
import numpy as np


def py_compute_optimal_transport(
    M: np.ndarray, r: np.ndarray, c: np.ndarray, lam: float, epsilon=1e-6
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

    col_sum = tf.reduce_sum(Pij, axis=2)  # (batch_size, num_keypoints)
    u = tf.reshape(a_vec / col_sum, [batch_size, -1, 1])
    Pij = Pij * u

    row_sum = tf.reduce_sum(Pij, axis=1)  # (batch_size, num_keypoints)
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


def assignment_log_likelihood(Pij, matches):
    matches = tf.convert_to_tensor(matches)
    batch_size, num_keypoints, num_keypoint_b = Pij.shape.as_list()
    assert num_keypoints == num_keypoint_b, "Assumption assertion"
    assert matches.shape.as_list()[-1] == 2
    assert matches.dtype in [tf.int32, tf.int64]

    dustbin_index_match = tf.ones_like(matches) * (num_keypoints - 1)
    matches = tf.where(tf.equal(matches, -1), x=dustbin_index_match, y=matches)

    pij_matches = tf.gather_nd(Pij, matches, batch_dims=1)
    pij_matches = tf.clip_by_value(pij_matches, 1e-7, 1.0)
    loss = -tf.math.log(pij_matches)
    loss = tf.reduce_mean(loss, -1)
    return tf.reduce_mean(loss)


def assignment_log_likelihood_from_matrix(pij_labels, pij_predicted, scale=1.0):
    pij = pij_predicted
    pij = tf.clip_by_value(pij, 1e-7, 1.0)
    log_loss = -pij_labels * tf.math.log(pij)
    log_loss = tf.reduce_sum(log_loss, axis=[1, 2])
    return tf.reduce_mean(log_loss) * scale


def get_assignment_loss(scale: float):
    return lambda targets, predicted: assignment_log_likelihood_from_matrix(
        targets, predicted, scale=scale
    )


class SinkhornKnoppLayer(tf.keras.layers.Layer):
    def __init__(self, lam: float = 5.0, num_steps: int = 100):
        super(SinkhornKnoppLayer, self).__init__()
        self.lam = lam
        self.num_steps = num_steps

    def normalize_features(self, features):
        return tf.math.l2_normalize(features, axis=-1)

    def compute_score_matrix(self, fA, fB):
        # (batch_size, num_keypoints, num_keypoints)
        Sij = tf.matmul(fA, fB, transpose_b=True)
        return Sij

    def compute_optimal_transport(self, Sij, a_vec, b_vec):
        Cij = -Sij  # convert scores matrix to cost matrix
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
        Pij_0 = alpha_beta * tf.exp(-self.lam * Cij)
        Pij_0 = normalize_pij(Pij_0)
        return sinkhorn_step(Pij_0, a_vec, b_vec)

    def call(self, fA, fB):
        _, num_keypoints, _ = fA.shape.as_list()
        assert fA.shape == fB.shape
        fA = self.normalize_features(fA)
        fB = self.normalize_features(fB)
        # (batch_size, num_keypoints, num_keypoints)
        Sij = self.compute_score_matrix(fA, fB)
        a_vec = tf.ones([num_keypoints])
        b_vec = tf.ones([num_keypoints])
        return self.compute_optimal_transport(Sij, a_vec, b_vec), Sij


class AugmentedSinkhornKnoppLayer(SinkhornKnoppLayer):
    def __init__(self, lam: float = 5.0, num_steps: int = 100, dustbin_init=0.0):
        super(AugmentedSinkhornKnoppLayer, self).__init__(lam=lam, num_steps=num_steps)
        self.dustbin_variable = tf.Variable(dustbin_init)

    def call(self, fA, fB):
        _, num_keypoints, latent_size = fA.shape.as_list()
        batch_size = tf.shape(fA)[0]
        assert fA.shape[1:] == fB.shape[1:]
        fA = self.normalize_features(fA)
        fB = self.normalize_features(fB)
        Sij = self.compute_score_matrix(fA, fB)

        row = self.dustbin_variable * tf.ones([batch_size, 1, num_keypoints])
        col = self.dustbin_variable * tf.ones([batch_size, num_keypoints + 1, 1])
        Sij_aug = tf.concat([Sij, row], axis=1)
        Sij_aug = tf.concat([Sij_aug, col], axis=2)  # [batch_size, nk +1, nk + 1]

        a_vec = np.array([[1.0] * num_keypoints + [num_keypoints]])
        b_vec = np.array([[1.0] * num_keypoints + [num_keypoints]])
        a_vec = tf.convert_to_tensor(a_vec)
        b_vec = tf.convert_to_tensor(b_vec)
        return self.compute_optimal_transport(Sij_aug, a_vec, b_vec), Sij
