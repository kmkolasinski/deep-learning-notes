import unittest

import tensorflow as tf
import numpy as np
import superglue.layers as layers

K = tf.keras.backend
keras = tf.keras


class TestMLP(tf.test.TestCase):
    def test_mlp(self):
        mlp = layers.MLP(depths=[32, 64, 128], output_dim=256)
        x = tf.random.uniform([32, 512, 2])
        outputs = mlp(x, training=True)
        self.assertEqual(outputs.shape, [32, 512, 256])
