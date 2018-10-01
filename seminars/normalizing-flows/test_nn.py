import real_nvp_nn as nn
import tensorflow as tf
import numpy as np
from tensorflow.contrib import framework as tf_framework


class TestLayers(tf.test.TestCase):

    def test_input_layer(self):
        images = np.random.rand(8, 32, 32, 1)
        images = tf.to_float(images)

        x, logdet, z = nn.InputLayer(images)

        self.assertEquals(images, x)
        self.assertEquals(logdet.shape.as_list(), [8])
        self.assertEqual(z, None)

    def forward_inverse(self, flow: nn.FlowLayer, inputs: nn.FlowData):
        x_input = inputs
        y = flow(x_input, forward=True)
        x_rec = flow(y, forward=False)

        with self.test_session() as sess:
            for c, cprim in zip(x_rec, x_input):
                if type(c) == tf.Tensor and type(cprim) == tf.Tensor:
                    c_np, cprim_np = sess.run([c, cprim])
                    self.assertAllClose(c_np, cprim_np)

    def test_squeezing_layer_conv(self):
        images = np.random.rand(8, 32, 32, 1)
        images = tf.to_float(images)

        flow = nn.InputLayer(images)

        layer = nn.SqueezingLayer()
        self.forward_inverse(layer, flow)

        new_flow = layer(flow, forward=True)
        x, logdet, z = new_flow

        self.assertEquals(logdet.shape.as_list(), [8])
        self.assertEquals([8, 16, 16, 4], x.shape.as_list())

    def test_factor_out_layer_conv(self):
        images_np = np.random.rand(8, 32, 32, 16)
        images = tf.to_float(images_np)

        flow = nn.InputLayer(images)
        layer = nn.FactorOutLayer()
        self.forward_inverse(layer, flow)
        with tf_framework.arg_scope([nn.FlowLayer.__call__], forward=True):
            new_flow = layer(flow)
        x, logdet, z = new_flow

        self.assertEquals([8, 32, 32, 8], x.shape.as_list())
        self.assertEquals([8, 32, 32, 8], z.shape.as_list())

        with self.test_session() as sess:
            new_flow_np = sess.run(new_flow)
            # x
            self.assertAllClose(new_flow_np[0], images_np[:, :, :, 8:])
            # z
            self.assertAllClose(new_flow_np[2], images_np[:, :, :, :8])
            # logdet
            self.assertAllClose(new_flow_np[1], np.zeros_like(new_flow_np[1]))

    def test_combine_squeeze_and_factor_layers_conv(self):

        images_np = np.random.rand(8, 32, 32, 1)
        images = tf.to_float(images_np)

        flow = nn.InputLayer(images)

        layers = [
            nn.SqueezingLayer(),  # x=[8, 16, 16, 4]
            nn.FactorOutLayer(),  # x=[8, 16, 16, 2]
            nn.SqueezingLayer(),  # x=[8, 8, 8, 8]
            nn.FactorOutLayer(),  # x=[8, 8, 8, 4] z=[8, 8, 8, 12]
        ]

        chain = nn.ChainLayer(layers=layers)
        print()
        with tf_framework.arg_scope([nn.FlowLayer.__call__], forward=True):
            new_flow = chain(flow)
            with self.test_session() as sess:
                x, logdet, z = sess.run(new_flow)
                self.assertEquals(x.shape, (8, 8, 8, 4))
                self.assertEquals(z.shape, (8, 8, 8, 12))

        self.forward_inverse(chain, flow)