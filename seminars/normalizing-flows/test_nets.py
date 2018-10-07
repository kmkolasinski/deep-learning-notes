import tensorflow as tf
import flow_layers as fl
import nets as nets
import numpy as np

K = tf.keras.backend
keras = tf.keras


class TestSimpleFlow(tf.test.TestCase):
    def forward_inverse(
        self,
        sess: tf.Session,
        flow: fl.FlowLayer,
        inputs: fl.FlowData,
        feed_dict=None,
        atol: float = 1e-6,
    ):
        x_input = inputs
        y = flow(x_input, forward=True)
        x_rec = flow(y, forward=False)

        for c, cprim in zip(x_rec, x_input):
            if type(c) == tf.Tensor and type(cprim) == tf.Tensor:
                c_np, cprim_np = sess.run([c, cprim], feed_dict=feed_dict)
                self.assertAllClose(c_np, cprim_np, atol=atol)

    def test_create_simple_flow(self):
        np.random.seed(642201)

        images = tf.placeholder(tf.float32, [16, 32, 32, 1])
        layers, actnorm_layers = nets.create_simple_flow(
            num_steps=2,
            num_scales=4,
            num_bits=8
        )
        flow = fl.InputLayer(images)
        model_flow = fl.ChainLayer(layers)
        output_flow = model_flow(flow, forward=True)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            for actnorm_layer in actnorm_layers:
                init_op = actnorm_layer.get_ddi_init_ops(10)
                noise = np.random.rand(16, 32, 32, 1)
                # fit actnorms to certain noise
                for i in range(30):
                    sess.run(init_op, feed_dict={images: noise})

                actnorm_flow = actnorm_layer._forward_outputs[0]
                normed_x = sess.run(actnorm_flow[0], feed_dict={images: noise})
                nc = normed_x.shape[-1]

                self.assertAllClose(
                    np.var(normed_x.reshape([-1, nc]), axis=0), [1.0] * nc, atol=0.1
                )
                self.assertAllClose(
                    np.mean(normed_x.reshape([-1, nc]), axis=0), [0.0] * nc, atol=0.1
                )

            output_flow_np = sess.run(
                output_flow, feed_dict={images: np.random.rand(16, 32, 32, 1)}
            )

            y, logdet, z = output_flow_np

            self.assertEqual(
                np.prod(y.shape) + np.prod(z.shape), np.prod([16, 32, 32, 1])
            )
            self.assertTrue(np.max(np.abs(y)) < 15.0)
            self.forward_inverse(
                sess,
                model_flow,
                flow,
                atol=0.01,
                feed_dict={images: np.random.rand(16, 32, 32, 1)},
            )

    def test_initialize_actnorms(self):

        np.random.seed(642201)
        images_ph = tf.placeholder(tf.float32, [16, 16, 16, 1])

        layers, actnorm_layers = nets.create_simple_flow(num_steps=1, num_scales=3)
        flow = fl.InputLayer(images_ph)
        model_flow = fl.ChainLayer(layers)
        output_flow = model_flow(flow, forward=True)

        noise = np.random.rand(16, 16, 16, 1)

        def feed_dict_fn():
            return {images_ph: noise}

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            nets.initialize_actnorms(
                sess,
                feed_dict_fn=feed_dict_fn,
                actnorm_layers=actnorm_layers,
                num_steps=50,
            )

            for actnorm_layer in actnorm_layers:

                actnorm_flow = actnorm_layer._forward_outputs[0]
                normed_x = sess.run(actnorm_flow[0], feed_dict={images_ph: noise})
                nc = normed_x.shape[-1]

                self.assertAllClose(
                    np.var(normed_x.reshape([-1, nc]), axis=0), [1.0] * nc, atol=0.1
                )
                self.assertAllClose(
                    np.mean(normed_x.reshape([-1, nc]), axis=0), [0.0] * nc, atol=0.1
                )

    def test_openai_template(self):

        np.random.seed(642201)
        images_fwd = tf.placeholder(tf.float32, [8, 16, 16, 24])
        images_bwd = tf.placeholder(tf.float32, [8, 16, 16, 24])
        template = nets.OpenAITemplate(width=32)
        block_fn = template.create_template_fn("Block")
        shift, scale = block_fn(images_fwd)

        all_trainable_fwd = tf.trainable_variables()

        shift_bwd, scale_bwd = block_fn(images_bwd)

        all_trainable_bwd = tf.trainable_variables()

        self.assertEqual(shift.shape.as_list(), [8, 16, 16, 24])
        self.assertEqual(scale.shape.as_list(), [8, 16, 16, 24])

        self.assertEqual(all_trainable_fwd, all_trainable_bwd)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            images = np.random.rand(8, 16, 16, 24)
            shift, scale = sess.run([shift, scale], feed_dict={
                images_fwd: images
            })

            shift_bwd, scale_bwd = sess.run([shift_bwd, scale_bwd], feed_dict={
                images_bwd: images
            })
            print(all_trainable_bwd)
            self.assertAllClose(shift, shift_bwd)
            self.assertAllClose(scale, scale_bwd)
