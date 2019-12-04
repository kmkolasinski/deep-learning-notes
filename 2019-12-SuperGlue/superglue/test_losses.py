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
        self.assertAllClose(P[0].sum(1), a[0].numpy() / 10)
        self.assertAllClose(P[0].sum(0), b[0].numpy() / 20)
        self.assertAllClose(P[0], P[1])
        self.assertAllClose(P[0], P[2])

    def test_batch_compute_optimal_transport(self):
        lam = 1
        M = tf.ones([3, 5, 5])
        a = tf.constant([[1, 1, 1, 1, 6]], dtype=tf.float32)
        b = tf.constant([[1, 1, 1, 1, 16]], dtype=tf.float32)
        P, dist = losses.batch_compute_optimal_transport(
            M, a, b, lam=lam, num_steps=201
        )
        P1 = losses.sinkhorn_step(P, a, b)
        print(P1)
        P = P.numpy()

        self.assertAllClose(dist, [1.0, 1.0, 1.0])
        self.assertAllClose(P[0].sum(1), a[0].numpy() / 10)
        self.assertAllClose(P[0].sum(0), b[0].numpy() / 20)
        self.assertAllClose(P[0], P[1])
        self.assertAllClose(P[0], P[2])

    def test_sinkhorn_knopp_layer(self):

        sk_layer = losses.SuperGlueSinkhornKnoppLayer(lam=5.0, num_steps=200)

        with tf.GradientTape() as tape:
            sA = tf.random.normal([3, 5, 7])
            sB = tf.random.normal([3, 5, 7])
            tape.watch([sA, sB])
            Pij = sk_layer(sA, sB)
            loss = - tf.math.log(Pij[:, 0, 0])
            gradients = tape.gradient(loss, [sk_layer.weights[0], sA, sB])
        Pij = Pij.numpy()
        print(Pij[0].sum(0))
        self.assertEqual(Pij.shape, (3, 6, 6))

    def test_sinkhorn_knopp_layer_static(self):
        sk_layer = losses.SuperGlueSinkhornKnoppLayer(lam=5.0, num_steps=100)

        @tf.function
        def step():
            with tf.GradientTape() as tape:
                sA = tf.random.normal([3, 5, 7])
                sB = tf.random.normal([3, 5, 7])
                tape.watch([sA, sB])
                Pij = sk_layer(sA, sB)
                loss = - tf.math.log(Pij[:, 0, 0])
                gradients = tape.gradient(loss, sk_layer.weights)
            return loss, gradients

        loss, grads = step()
        print(loss)
        loss, grads = step()
        print(loss)

