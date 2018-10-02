import real_nvp_nn as nn
import tensorflow as tf
import numpy as np
from tensorflow.contrib import framework as tf_framework


class TestLayers(tf.test.TestCase):
    def test_input_layer(self):
        images = np.random.rand(8, 32, 32, 1)
        images = tf.to_float(images)

        x, logdet, z = nn.InputLayer(images)

        self.assertEqual(images, x)
        self.assertEqual(logdet.shape.as_list(), [8])
        self.assertEqual(z, None)

    def forward_inverse(
        self,
        flow: nn.FlowLayer,
        inputs: nn.FlowData,
        feed_dict=None,
        atol: float = 1e-6,
    ):
        x_input = inputs
        y = flow(x_input, forward=True)
        x_rec = flow(y, forward=False)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            for c, cprim in zip(x_rec, x_input):
                if type(c) == tf.Tensor and type(cprim) == tf.Tensor:
                    c_np, cprim_np = sess.run([c, cprim], feed_dict=feed_dict)
                    self.assertAllClose(c_np, cprim_np, atol=atol)

    def test_logitify_layer_conv(self):
        np.random.seed(52321)
        images_np = np.random.rand(8, 32, 32, 1)
        images = tf.to_float(images_np)

        flow = nn.InputLayer(images)

        layer = nn.LogitifyImage()
        self.forward_inverse(layer, flow, atol=0.01)

        x, logdet, z = flow
        logdet += 10.0
        flow = x, logdet, z

        new_flow = layer(flow, forward=True)
        flow_rec = layer(new_flow, forward=False)
        x, logdet, z = new_flow
        x_rec, logdet_rec, z = flow_rec

        self.assertEqual(z, None)
        self.assertEqual(x.shape.as_list(), [8, 32, 32, 1])
        self.assertEqual(logdet.shape.as_list(), [8])
        self.assertEqual(x_rec.shape.as_list(), [8, 32, 32, 1])
        self.assertEqual(logdet_rec.shape.as_list(), [8])

        with self.test_session() as sess:
            _ = sess.run([x, logdet])
            x_rec, logdet_rec = sess.run([x_rec, logdet_rec])
            self.assertAllClose(logdet_rec, [10.0] * 8, atol=0.01)
            self.assertAllClose(images_np, x_rec, atol=0.01)

    def test_squeezing_layer_conv(self):
        images = np.random.rand(8, 32, 32, 1)
        images = tf.to_float(images)

        flow = nn.InputLayer(images)

        layer = nn.SqueezingLayer()
        self.forward_inverse(layer, flow)

        new_flow = layer(flow, forward=True)
        x, logdet, z = new_flow

        self.assertEqual(logdet.shape.as_list(), [8])
        self.assertEqual([8, 16, 16, 4], x.shape.as_list())

    def test_factor_out_layer_conv(self):
        images_np = np.random.rand(8, 32, 32, 16)
        images = tf.to_float(images_np)

        flow = nn.InputLayer(images)
        layer = nn.FactorOutLayer()
        self.forward_inverse(layer, flow)
        with tf_framework.arg_scope([nn.FlowLayer.__call__], forward=True):
            new_flow = layer(flow)
        x, logdet, z = new_flow

        self.assertEqual([8, 32, 32, 8], x.shape.as_list())
        self.assertEqual([8, 32, 32, 8], z.shape.as_list())

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
        # in comments are output shapes
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
                self.assertEqual(x.shape, (8, 8, 8, 4))
                self.assertEqual(z.shape, (8, 8, 8, 12))

        self.forward_inverse(chain, flow)

    def test_actnorm_bias_conv(self):

        images_np = np.random.rand(8, 32, 32, 3)
        images = tf.to_float(images_np)

        flow = nn.InputLayer(images)

        layer = nn.ActnormBiasLayer()
        new_flow = layer(flow, forward=True)
        x, logdet, z = new_flow

        self.assertEqual(z, None)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            bias = sess.run(layer._bias_t)
            x, logdet = sess.run([x, logdet])
            self.assertEqual(x.shape, images_np.shape)
            self.assertEqual(bias.shape, (1, 1, 1, 3))
            self.assertAllClose(np.sum(bias ** 2), 0)

        self.forward_inverse(layer, flow)

    def test_actnorm_bias_init_conv(self):
        images_np = np.random.rand(8, 32, 32, 3)
        images = tf.to_float(images_np)

        flow = nn.InputLayer(images)

        layer = nn.ActnormBiasLayer()

        with self.assertRaises(AssertionError):
            layer.get_ddi_init_ops()

        new_flow = layer(flow, forward=True)
        x, logdet, z = new_flow

        init_ops = layer.get_ddi_init_ops()
        self.assertEqual(z, None)

        with self.test_session() as sess:
            # initialize network
            sess.run(tf.global_variables_initializer())
            sess.run(init_ops)
            x, logdet = sess.run([x, logdet])
            bias = sess.run(layer._bias_t)

            self.assertEqual(x.shape, images_np.shape)
            self.assertEqual(bias.shape, (1, 1, 1, 3))
            self.assertGreater(np.sum(bias ** 2), 0)
            # check mean after passing act norm

            self.assertAllClose(np.mean(x.reshape([-1, 3]), axis=0), [0.0, 0.0, 0.0])

        self.forward_inverse(layer, flow)

    def test_actnorm_bias_init_conv_iter(self):

        images_ph = tf.placeholder(tf.float32, shape=[8, 32, 32, 3])

        flow = nn.InputLayer(images_ph)

        layer = nn.ActnormBiasLayer()

        new_flow = layer(flow, forward=True)
        x, logdet, z = new_flow
        init_ops = layer.get_ddi_init_ops(num_init_iterations=100)

        with self.test_session() as sess:
            # initialize network
            sess.run(tf.global_variables_initializer())
            for i in range(200):
                sess.run(init_ops, feed_dict={images_ph: np.random.rand(8, 32, 32, 3)})
                # print(sess.run(layer._bias_t))
            x_np_values = []
            for i in range(20):
                x_np, logdet_np = sess.run(
                    [x, logdet], feed_dict={images_ph: np.random.rand(8, 32, 32, 3)}
                )

                x_np_values.append(x_np)

            x_np_values = np.array(x_np_values).mean(0)

            self.assertEqual(x.shape, x_np_values.shape)

            self.assertAllClose(
                np.mean(x_np_values.reshape([-1, 3]), axis=0),
                [0.0, 0.0, 0.0],
                atol=0.05,
            )

        self.forward_inverse(
            layer, flow, feed_dict={images_ph: np.random.rand(8, 32, 32, 3)}
        )

    def test_actnorm_scale_conv(self):

        images_np = np.random.rand(8, 32, 32, 3)
        images = tf.to_float(images_np)

        flow = nn.InputLayer(images)

        layer = nn.ActnormScaleLayer()
        new_flow = layer(flow, forward=True)
        x, logdet, z = new_flow

        self.assertEqual(z, None)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            log_scale = sess.run(layer._log_scale_t)
            x, logdet = sess.run([x, logdet])

            self.assertEqual(x.shape, images_np.shape)
            self.assertEqual(log_scale.shape, (1, 1, 1, 3))
            self.assertEqual(np.sum(log_scale ** 2), 0)
            # zero initialization
            self.assertAllClose(logdet, [0.0] * 8)

        self.forward_inverse(layer, flow)

    def test_actnorm_scale_init_conv(self):
        np.random.seed(52321)
        images_np = np.random.rand(8, 32, 32, 3)
        images = tf.to_float(images_np)

        flow = nn.InputLayer(images)

        layer = nn.ActnormScaleLayer()

        with self.assertRaises(AssertionError):
            layer.get_ddi_init_ops()

        new_flow = layer(flow, forward=True)
        x, logdet, z = new_flow

        init_ops = layer.get_ddi_init_ops()
        self.assertEqual(z, None)

        with self.test_session() as sess:
            # initialize network
            sess.run(tf.global_variables_initializer())
            sess.run(init_ops)
            x, logdet = sess.run([x, logdet])
            log_scale = sess.run(layer._log_scale_t)

            self.assertEqual(x.shape, images_np.shape)
            self.assertEqual(log_scale.shape, (1, 1, 1, 3))
            self.assertGreater(np.sum(log_scale ** 2), 0)
            self.assertGreater(np.sum(logdet ** 2), 0)
            # check var after passing act norm
            self.assertAllClose(
                np.var(x.reshape([-1, 3]), axis=0), [1.0, 1.0, 1.0], atol=0.001
            )

        self.forward_inverse(layer, flow)

    def test_actnorm_scale_init_conv_scale(self):
        np.random.seed(52321)
        images_np = np.random.rand(8, 32, 32, 3)
        images = tf.to_float(images_np)

        flow = nn.InputLayer(images)

        layer = nn.ActnormScaleLayer(scale=np.sqrt(2))

        with self.assertRaises(AssertionError):
            layer.get_ddi_init_ops()

        new_flow = layer(flow, forward=True)
        x, logdet, z = new_flow

        init_ops = layer.get_ddi_init_ops()
        self.assertEqual(z, None)

        with self.test_session() as sess:
            # initialize network
            sess.run(tf.global_variables_initializer())
            sess.run(init_ops)
            x, logdet = sess.run([x, logdet])
            log_scale = sess.run(layer._log_scale_t)

            self.assertEqual(x.shape, images_np.shape)
            self.assertEqual(log_scale.shape, (1, 1, 1, 3))
            self.assertGreater(np.sum(log_scale ** 2), 0)
            # check var after passing act norm
            self.assertAllClose(
                np.var(x.reshape([-1, 3]), axis=0), [2.0, 2.0, 2.0], atol=0.001
            )

        self.forward_inverse(layer, flow)

    def test_actnorm_scale_init_conv_iter(self):
        np.random.seed(52321)
        images_ph = tf.placeholder(tf.float32, shape=[8, 32, 32, 3])

        flow = nn.InputLayer(images_ph)

        layer = nn.ActnormScaleLayer(scale=np.sqrt(np.pi))

        new_flow = layer(flow, forward=True)
        x, logdet, z = new_flow
        init_ops = layer.get_ddi_init_ops(num_init_iterations=50)

        with self.test_session() as sess:
            # initialize network
            sess.run(tf.global_variables_initializer())
            for i in range(200):
                sess.run(init_ops, feed_dict={images_ph: np.random.rand(8, 32, 32, 3)})

            for i in range(5):
                x_np, logdet_np = sess.run(
                    [x, logdet], feed_dict={images_ph: np.random.rand(8, 32, 32, 3)}
                )

                self.assertEqual(x.shape, x_np.shape)
                self.assertAllClose(
                    np.var(x_np.reshape([-1, 3]), axis=0),
                    [np.pi, np.pi, np.pi],
                    atol=0.1,
                )

        self.forward_inverse(
            layer, flow, feed_dict={images_ph: np.random.rand(8, 32, 32, 3)}
        )

    def test_actnorm_conv(self):

        images_np = np.random.rand(8, 32, 32, 3)
        images = tf.to_float(images_np)

        flow = nn.InputLayer(images)

        layer = nn.ActnormLayer()
        new_flow = layer(flow, forward=True)
        x, logdet, z = new_flow

        self.assertEqual(z, None)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            x, logdet = sess.run([x, logdet])
            self.assertEqual(x.shape, images_np.shape)

        self.forward_inverse(layer, flow)

    def test_actnorm_init_conv(self):
        np.random.seed(52321)
        images_np = np.random.rand(8, 32, 32, 3)
        images = tf.to_float(images_np)

        flow = nn.InputLayer(images)

        layer = nn.ActnormLayer(scale=np.sqrt(np.pi))

        with self.assertRaises(AssertionError):
            layer.get_ddi_init_ops()

        new_flow = layer(flow, forward=True)
        x, logdet, z = new_flow

        init_ops = layer.get_ddi_init_ops()
        self.assertEqual(z, None)

        with self.test_session() as sess:
            # initialize network
            sess.run(tf.global_variables_initializer())
            sess.run(init_ops)
            x, logdet = sess.run([x, logdet])

            self.assertEqual(x.shape, images_np.shape)
            # check var after passing act norm
            self.assertAllClose(
                np.var(x.reshape([-1, 3]), axis=0), [np.pi] * 3, atol=0.01
            )
            self.assertAllClose(
                np.mean(x.reshape([-1, 3]), axis=0), [0.0] * 3, atol=0.01
            )

        self.forward_inverse(layer, flow)

    def test_actnorm_init_conv_iter(self):
        np.random.seed(52321)
        images_ph = tf.placeholder(tf.float32, shape=[8, 32, 32, 3])

        flow = nn.InputLayer(images_ph)

        layer = nn.ActnormLayer(scale=np.sqrt(np.pi))

        new_flow = layer(flow, forward=True)
        x, logdet, z = new_flow
        init_ops = layer.get_ddi_init_ops(num_init_iterations=50)

        with self.test_session() as sess:
            # initialize network
            sess.run(tf.global_variables_initializer())
            for i in range(200):
                sess.run(init_ops, feed_dict={images_ph: np.random.rand(8, 32, 32, 3)})

            for i in range(5):
                x_np, logdet_np = sess.run(
                    [x, logdet], feed_dict={images_ph: np.random.rand(8, 32, 32, 3)}
                )
                self.assertEqual(x.shape, x_np.shape)
                self.assertAllClose(
                    np.var(x_np.reshape([-1, 3]), axis=0), [np.pi] * 3, atol=0.1
                )
                self.assertAllClose(
                    np.mean(x_np.reshape([-1, 3]), axis=0), [0.0] * 3, atol=0.1
                )

        self.forward_inverse(
            layer, flow, feed_dict={images_ph: np.random.rand(8, 32, 32, 3)}
        )
