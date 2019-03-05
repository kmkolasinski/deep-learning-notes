import numpy as np
import tensorflow as tf

import neural_ode
import cnf

tf.enable_eager_execution()
K = tf.keras.backend
keras = tf.keras


class TestCNF(tf.test.TestCase):
    def test_hypernet(self):

        net = cnf.HyperNet(input_dim=2, hidden_dim=32, n_ensemble=16)

        output = net(tf.to_float(1.0))
        output_shapes = net.compute_output_shape([])
        for o, exp_shape in zip(output, output_shapes):
            self.assertEqual(o.shape.as_list(), exp_shape)

    def test_cnf(self):
        net = cnf.CNF(input_dim=2, hidden_dim=32, n_ensemble=16)
        x = tf.random_normal([32, 3])
        output = net([tf.to_float(1.0), x])
        self.assertEqual(output.shape.as_list(), [32, 3])

    def test_cnf_integration(self):

        p0 = tf.distributions.Normal(loc=[0.0, 0.0], scale=[1.0, 1.0])
        net = cnf.CNF(input_dim=2, hidden_dim=32, n_ensemble=16)
        ode = neural_ode.NeuralODE(
            model=net, t=np.linspace(1, 0.0, 10), solver=neural_ode.rk4_step
        )

        logdet0 = tf.zeros([32, 1])
        x0 = tf.random_normal([32, 2])
        h0 = tf.concat([x0, logdet0], axis=1)

        hN = ode.forward(inputs=h0)
        with tf.GradientTape() as g:
            g.watch(hN)
            xN, logdetN = hN[:, :2], hN[:, 2]
            # L = log(p(zN))
            mle = tf.reduce_sum(p0.log_prob(xN), -1)
            # loss to minimize
            loss = -tf.reduce_mean(mle - logdetN)

        dloss = g.gradient(loss, hN)
        h0_rec, dLdh0, dLdW = ode.backward(hN, dloss)
        self.assertAllClose(h0_rec, h0)
