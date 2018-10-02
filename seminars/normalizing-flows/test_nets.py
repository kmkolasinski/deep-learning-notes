from pprint import pprint

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

        images = tf.placeholder(tf.float32, [16, 32, 32, 1])

        layers = nets.create_simple_flow(num_steps=2, num_scales=4)
        flow = fl.InputLayer(images)
        model_flow = fl.ChainLayer(layers)

        output_flow = model_flow(flow, forward=True)

        init_ops = []

        for layer in model_flow._layers:
            if type(layer) == fl.ChainLayer:
                for c_layer in layer._layers:
                    if type(c_layer) == fl.ChainLayer:
                        for c2_layer in c_layer._layers:
                            if type(c2_layer) == fl.ActnormLayer:
                                update_op = c2_layer.get_ddi_init_ops(0)
                                init_ops.append(update_op)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(1):
                sess.run(init_ops, feed_dict={
                        images: np.random.rand(16, 32, 32, 1)
                })

            output_flow_np = sess.run(
                output_flow, feed_dict={
                    images: np.random.rand(16, 32, 32, 1)
                }
            )
            print(output_flow_np[0])

            self.forward_inverse(sess, model_flow, flow, atol=0.01, feed_dict={images: np.random.rand(16, 32, 32, 1)})