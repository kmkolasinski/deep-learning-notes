import os
import tensorflow as tf
import numpy as np
import superglue.losses as losses

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

K = tf.keras.backend
keras = tf.keras


class TestSinkhornKnoppLayer(tf.test.TestCase):
    def test_normalize_pij(self):
        lam = 5
        M = tf.random.normal([4, 15, 15])
        P = tf.exp(-lam * M)
        P = losses.normalize_pij(P)
        P: np.ndarray = P.numpy()
        self.assertAllClose(P.reshape([4, -1]).sum(axis=-1), [1, 1, 1, 1])

    def test_sinkhorn_step(self):
        lam = 5
        M = tf.ones([3, 5, 5])
        P = tf.exp(-lam * M)
        P = losses.normalize_pij(P)
        a = tf.constant([[1, 1, 1, 1, 6]], dtype=tf.float32)
        b = tf.constant([[1, 1, 1, 1, 16]], dtype=tf.float32)
        for i in range(200):
            P = losses.sinkhorn_step(P, a, b)
        P = P.numpy()

        self.assertAllClose(P[0].sum(0), b[0].numpy())
        self.assertAllClose(P[0], P[1])
        self.assertAllClose(P[0], P[2])

    def test_batch_compute_optimal_transport(self):
        lam = 1
        M = tf.random.normal([1, 5, 5])
        a = tf.constant([[1, 1, 1, 1, 6]], dtype=tf.float32)
        b = tf.constant([[1, 1, 1, 1, 16]], dtype=tf.float32)
        P, dist = losses.batch_compute_optimal_transport(
            M, a, b, lam=lam, num_steps=100
        )
        Pnp, _ = losses.py_compute_optimal_transport(
            M[0].numpy(), a[0].numpy(), b[0].numpy(), lam=lam
        )
        self.assertAllClose(Pnp, P[0].numpy())
        # converged solution should not change with one additional step
        Pnext = losses.sinkhorn_step(P, a, b)
        self.assertAllClose(P.numpy(), Pnext.numpy())

    def test_sinkhorn_knopp_layer(self):

        sk_layer = losses.SinkhornKnoppLayer(lam=5.0, num_steps=200)
        fA = tf.random.normal([3, 5, 7])
        fB = tf.random.normal([3, 5, 7])
        Pij, Sij = sk_layer(fA, fB)
        self.assertEqual(Pij.shape, [3, 5, 5])
        a_vec = np.ones([5])
        b_vec = np.ones([5])
        Pij_np, _ = losses.py_compute_optimal_transport(- Sij[0].numpy(), a_vec, b_vec, lam=5)
        self.assertAllClose(Pij_np, Pij[0].numpy(), atol=1e-6)

    def test_super_glue_sinkhorn_knopp_layer(self):

        sk_layer = losses.SuperGlueSinkhornKnoppLayer(lam=5.0, num_steps=200)

        with tf.GradientTape() as tape:
            sA = tf.random.normal([3, 5, 7])
            sB = tf.random.normal([3, 5, 7])
            tape.watch([sA, sB])
            Pij, _ = sk_layer(sA, sB)
            loss = - tf.math.log(Pij[:, 0, 0])
            gradients = tape.gradient(loss, [sk_layer.weights[0], sA, sB])

        self.assertAllEqual(np.abs(gradients[1].numpy()) > 0, np.ones_like(sA))
        self.assertAllEqual(np.abs(gradients[2].numpy()) > 0, np.ones_like(sB))
        Pij = Pij.numpy()
        self.assertAllClose(Pij[0].sum(0), [1, 1, 1, 1, 1, 5])
        self.assertEqual(Pij.shape, (3, 6, 6))

    def test_super_glue_sinkhorn_knopp_layer_static(self):
        sk_layer = losses.SuperGlueSinkhornKnoppLayer(lam=5.0, num_steps=100)

        @tf.function
        def step():
            with tf.GradientTape() as tape:
                sA = tf.random.normal([3, 5, 7])
                sB = tf.random.normal([3, 5, 7])
                tape.watch([sA, sB])
                Pij, _ = sk_layer(sA, sB)
                loss = - tf.math.log(Pij[:, 0, 0])
                gradients = tape.gradient(loss, sk_layer.weights)
            return loss, gradients

        loss, grads = step()
        loss, grads = step()

