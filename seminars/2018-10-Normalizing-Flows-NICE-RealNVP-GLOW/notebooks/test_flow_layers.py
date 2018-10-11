from typing import Callable, Optional, Dict

import flow_layers as fl
import tensorflow as tf
import numpy as np
from tensorflow.contrib import framework as tf_framework
import tensorflow.contrib.layers as tf_layers
from tensorflow.python.ops import template as template_ops

K = tf.keras.backend
keras = tf.keras


def _shift_and_log_scale_fn_template(name):
    def _shift_and_log_scale_fn(x: tf.Tensor):
        shape = K.int_shape(x)
        num_channels = shape[3]
        # nn definition
        h = tf_layers.conv2d(x, num_outputs=num_channels, kernel_size=3)
        h = tf_layers.conv2d(h, num_outputs=num_channels // 2, kernel_size=3)
        # create shift and log_scale
        shift = tf_layers.conv2d(h, num_outputs=num_channels, kernel_size=3)
        log_scale = tf_layers.conv2d(
            h, num_outputs=num_channels, kernel_size=3, activation_fn=None
        )
        log_scale = tf.clip_by_value(log_scale, -15.0, 15.0)
        return shift, log_scale

    return template_ops.make_template(name, _shift_and_log_scale_fn)


class TestLayers(tf.test.TestCase):
    def test_input_layer(self):
        images = np.random.rand(8, 32, 32, 1)
        images = tf.to_float(images)

        x, logdet, z = fl.InputLayer(images)

        self.assertEqual(images, x)
        self.assertEqual(logdet.shape.as_list(), [8])
        self.assertEqual(z, None)

    def forward_inverse(
        self,
        flow: fl.FlowLayer,
        inputs: fl.FlowData,
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

    def try_to_train_identity_layer(
        self,
        layer: fl.FlowLayer,
        flow: fl.FlowData,
        feed_dict_fn: Optional[Callable[[], Dict[tf.Tensor, np.ndarray]]] = None,
        sess: Optional[tf.Session] = None,
        post_init_fn: Optional[Callable[[tf.Session], None]] = None,
    ):
        x, logdet, z = flow
        new_flow = layer(flow, forward=True, is_training=True)
        x_rec, logdet_rec, z_rec = new_flow
        loss = tf.losses.mean_squared_error(x, x_rec)
        opt = tf.train.MomentumOptimizer(0.1, 0.9)
        opt_op = opt.minimize(loss)

        sess = tf.Session() if sess is None else sess
        sess.run(tf.global_variables_initializer())
        if post_init_fn is not None:
            post_init_fn(sess)
        losses = []
        for i in range(50):
            if feed_dict_fn is not None:
                feed_dict = feed_dict_fn()
            else:
                feed_dict = None
            loss_np, _ = sess.run([loss, opt_op], feed_dict=feed_dict)
            losses.append(loss_np)

        self.assertGreater(losses[0], losses[-1])

    def test_logitify_layer_conv(self):
        np.random.seed(52321)
        images_np = np.random.rand(8, 32, 32, 1)
        images = tf.to_float(images_np)

        flow = fl.InputLayer(images)

        layer = fl.LogitifyImage()
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

    def test_quantize_image_layer_conv(self):
        np.random.seed(52321)
        images_np = np.random.rand(8, 32, 32, 3)
        images = tf.to_float(images_np)

        flow = fl.InputLayer(images)
        layer = fl.QuantizeImage(num_bits=8)
        self.forward_inverse(layer, flow, atol=1.5 / 256)

        new_flow = layer(flow, forward=True)
        flow_rec = layer(new_flow, forward=False)
        x, logdet, z = new_flow
        x_rec, logdet_rec, z = flow_rec

        self.assertEqual(z, None)
        self.assertEqual(x.shape.as_list(), [8, 32, 32, 3])
        self.assertEqual(x_rec.shape.as_list(), [8, 32, 32, 3])
        # less bits
        flow = fl.InputLayer(images)
        layer = fl.QuantizeImage(num_bits=5)
        self.forward_inverse(layer, flow, atol=1.5 / 32)

        layer = fl.QuantizeImage(num_bits=4)
        new_flow = layer(flow, forward=True)
        flow_rec = layer(new_flow, forward=False)

        with self.test_session() as sess:
            x_rec_uint8 = layer.to_uint8(flow_rec[0])
            self.assertEqual(x_rec_uint8.dtype, tf.uint8)
            x_rec_uint8 = sess.run(x_rec_uint8)
            self.assertAllGreaterEqual(x_rec_uint8, 0)
            self.assertAllLessEqual(x_rec_uint8, 255)
            self.assertEqual(np.unique(x_rec_uint8).shape, (2**4, ))

        with self.assertRaises(AssertionError):
            layer = fl.QuantizeImage(num_bits=4)
            self.forward_inverse(layer, flow, atol=1 / 32)

    def test_squeezing_layer_conv(self):
        images = np.random.rand(8, 32, 32, 1)
        images = tf.to_float(images)

        flow = fl.InputLayer(images)

        layer = fl.SqueezingLayer()
        self.forward_inverse(layer, flow)

        new_flow = layer(flow, forward=True)
        x, logdet, z = new_flow

        self.assertEqual(logdet.shape.as_list(), [8])
        self.assertEqual([8, 16, 16, 4], x.shape.as_list())

    def test_factor_out_layer_conv(self):
        images_np = np.random.rand(8, 32, 32, 16)
        images = tf.to_float(images_np)

        flow = fl.InputLayer(images)
        layer = fl.FactorOutLayer()
        self.forward_inverse(layer, flow)
        with tf_framework.arg_scope([fl.FlowLayer.__call__], forward=True):
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

        flow = fl.InputLayer(images)
        # in comments are output shapes
        layers = [
            fl.SqueezingLayer(),  # x=[8, 16, 16, 4]
            fl.FactorOutLayer(),  # x=[8, 16, 16, 2]
            fl.SqueezingLayer(),  # x=[8, 8, 8, 8]
            fl.FactorOutLayer(),  # x=[8, 8, 8, 4] z=[8, 8, 8, 12]
        ]

        chain = fl.ChainLayer(layers=layers)
        print()
        with tf_framework.arg_scope([fl.FlowLayer.__call__], forward=True):
            new_flow = chain(flow)
            with self.test_session() as sess:
                x, logdet, z = sess.run(new_flow)
                self.assertEqual(x.shape, (8, 8, 8, 4))
                self.assertEqual(z.shape, (8, 8, 8, 12))

        self.forward_inverse(chain, flow)

    def test_actnorm_bias_conv(self):

        images_np = np.random.rand(8, 32, 32, 3)
        images = tf.to_float(images_np)

        flow = fl.InputLayer(images)

        layer = fl.ActnormBiasLayer()
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

        flow = fl.InputLayer(images)

        layer = fl.ActnormBiasLayer()

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

        flow = fl.InputLayer(images_ph)

        layer = fl.ActnormBiasLayer()

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

        flow = fl.InputLayer(images)

        layer = fl.ActnormScaleLayer()
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

        flow = fl.InputLayer(images)

        layer = fl.ActnormScaleLayer()

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

        flow = fl.InputLayer(images)

        layer = fl.ActnormScaleLayer(scale=np.sqrt(2))

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

        flow = fl.InputLayer(images_ph)

        layer = fl.ActnormScaleLayer(scale=np.sqrt(np.pi))

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

        flow = fl.InputLayer(images)

        layer = fl.ActnormLayer()
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

        flow = fl.InputLayer(images)

        layer = fl.ActnormLayer(scale=np.sqrt(np.pi))

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

        flow = fl.InputLayer(images_ph)

        layer = fl.ActnormLayer(scale=np.sqrt(np.pi))

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

        def feed_dict_fn():
            return {images_ph: np.random.rand(8, 32, 32, 3)}

        def post_init_fn(sess):
            init_ops = layer.get_ddi_init_ops()
            sess.run(init_ops, {images_ph: np.random.rand(8, 32, 32, 3)})

        with tf.variable_scope("TestTrain"):
            layer = fl.ActnormLayer(scale=np.sqrt(np.pi))
            self.try_to_train_identity_layer(
                layer, flow, feed_dict_fn=feed_dict_fn, post_init_fn=post_init_fn
            )

    def test_invertible_conv1x1_no_lu_decomp(self):

        images_np = np.random.rand(8, 32, 32, 16)
        images = tf.to_float(images_np)

        flow = fl.InputLayer(images)

        layer = fl.InvertibleConv1x1Layer(use_lu_decomposition=False)
        new_flow = layer(flow, forward=True)
        x, logdet, z = new_flow

        self.assertEqual(z, None)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            x, logdet = sess.run([x, logdet])
            self.assertEqual(x.shape, images_np.shape)

        self.forward_inverse(layer, flow)

    def test_invertible_conv1x1_lu_decomp(self):

        images_np = np.random.rand(8, 32, 32, 16)
        images = tf.to_float(images_np)

        flow = fl.InputLayer(images)

        layer = fl.InvertibleConv1x1Layer(use_lu_decomposition=True)
        new_flow = layer(flow, forward=True)
        x, logdet, z = new_flow

        self.assertEqual(z, None)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            x, logdet = sess.run([x, logdet])
            self.assertEqual(x.shape, images_np.shape)

        self.forward_inverse(layer, flow)

    def test_invertible_conv1x1_learn_identity(self):

        images = tf.random_normal([8, 32, 32, 16])
        flow = fl.InputLayer(images)
        layer = fl.InvertibleConv1x1Layer(use_lu_decomposition=True)
        self.try_to_train_identity_layer(layer, flow)

        layer = fl.InvertibleConv1x1Layer(use_lu_decomposition=False)
        self.try_to_train_identity_layer(layer, flow)

    def test_simple_affine_coupling_layer(self):

        images_np = np.random.rand(8, 32, 32, 16)
        images = tf.to_float(images_np)
        flow = fl.InputLayer(images)
        layer = fl.AffineCouplingLayer(_shift_and_log_scale_fn_template("test"))
        new_flow = layer(flow, forward=True)
        x, logdet, z = new_flow

        self.assertEqual(z, None)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            x, logdet = sess.run([x, logdet])
            self.assertEqual(x.shape, images_np.shape)
            self.assertEqual(logdet.shape, (8,))

        self.forward_inverse(layer, flow)

        layer = fl.AffineCouplingLayer(_shift_and_log_scale_fn_template("train"))
        self.try_to_train_identity_layer(layer, flow)
