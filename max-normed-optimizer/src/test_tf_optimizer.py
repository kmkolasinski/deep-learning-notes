"""Tests for tf_optimizers"""

import numpy as np
import tensorflow as tf

import tf_optimizer as optimizers


class NormalizedSGDTest(tf.test.TestCase):

    def test_run(self):
        optimizer = optimizers.AdaptiveNormalizedSGD(
            lr=0.1
        )

        variable = tf.get_variable('var',
                                   initializer=np.zeros([10, 10], np.float32))

        loss = tf.reduce_mean(tf.square(variable - 1.0))

        update_op = optimizer.minimize(loss)
        loss_hist = []
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            for input_global_step in range(10):
                loss_np = sess.run([loss, update_op])
                loss_hist.append(loss_np)

        self.assertTrue(loss_hist[0] > loss_hist[-1])


class BarzilaiBorweinNormalizedSGDTest(tf.test.TestCase):

    def test_run(self):

        variable = tf.get_variable(
            'var', shape=[10, 10], dtype=tf.float32)

        loss = tf.reduce_mean(tf.square(variable - 1.0))

        optimizer = optimizers.BarzilaiBorweinNormalizedSGD(
            lr=0.1
        )

        update_op = optimizer.minimize(loss)
        loss_hist = []
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            for input_global_step in range(10):
                loss_np = sess.run([loss, update_op])
                loss_hist.append(loss_np)

        self.assertTrue(loss_hist[0] > loss_hist[-1])

    def test_current_step(self):

        variable = tf.get_variable(
            'var', shape=[10, 10], dtype=tf.float32)

        loss = tf.reduce_mean(tf.square(variable - 1.0))

        optimizer = optimizers.BarzilaiBorweinNormalizedSGD(
            lr=0.1, steps=20)

        update_op = optimizer.minimize(loss)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            for input_global_step in range(10):
                _ = sess.run([loss, update_op])
            self.assertEqual(sess.run(optimizer._current_step), 10)

    def test_current_step_and_steps(self):

        variable = tf.get_variable(
            'var', shape=[10, 10], dtype=tf.float32)

        loss = tf.reduce_mean(tf.square(variable - 1.0))

        optimizer = optimizers.BarzilaiBorweinNormalizedSGD(
            lr=0.1, steps=6)

        update_op = optimizer.minimize(loss)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            for _ in range(10):
                _ = sess.run([loss, update_op])
            self.assertEqual(sess.run(optimizer._current_step), 4)

            for _ in range(1):
                _ = sess.run([loss, update_op])
            self.assertEqual(sess.run(optimizer._current_step), 5)

            for _ in range(1):
                _ = sess.run([loss, update_op])
            self.assertEqual(sess.run(optimizer._current_step), 0)

    def test_grad_udpate(self):

        variable = tf.get_variable(
            'var',
            dtype=tf.float32, initializer=np.ones([10, 10]).astype(np.float32))

        loss = tf.reduce_mean(tf.square(variable - 1.0))

        optimizer = optimizers.BarzilaiBorweinNormalizedSGD(
            lr=0.1, steps=6)

        update_op = optimizer.minimize(loss)

        gk = optimizer.get_slot(variable, "gk")
        gk_old = optimizer.get_slot(variable, "gk_old")
        v_old = optimizer.get_slot(variable, "v_old")

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            for _ in range(5):
                _ = sess.run([loss, update_op])
            self.assertEqual(sess.run(optimizer._current_step), 5)
            gk_old_np = sess.run(gk_old)
            self.assertAllClose(gk_old_np.reshape([-1]), [0.0] * 100)
            v_old_np = sess.run(v_old)
            self.assertAllClose(v_old_np.reshape([-1]), [0.0] * 100)

            _ = sess.run([update_op])
            gk_old_np = sess.run(gk_old)
            self.assertTrue(np.abs(gk_old_np).mean() > 0)

            v_old_np = sess.run(v_old)

            self.assertTrue(np.abs(v_old_np).mean() > 0)

            gk_np = sess.run(gk)
            self.assertAllClose(gk_np.reshape([-1]), [0.0] * 100)

    def test_lr_update(self):

        variable = tf.get_variable(
            'var',
            dtype=tf.float32, initializer=np.zeros([10, 10]).astype(np.float32))

        loss = tf.reduce_mean(tf.square(variable - 1.0))
        lr_np = 0.5
        optimizer = optimizers.BarzilaiBorweinNormalizedSGD(
            lr=lr_np, steps=3, lr_max=5.0, lr_min=1e-6, lr_update=0.1,
            noise_amplitude=0.0)

        update_op = optimizer.minimize(loss)
        lr = optimizer._lr_variables[0]
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(100):
                loss_np, _ = sess.run([loss, update_op])
                lr_np = sess.run(lr)

            self.assertTrue(lr_np < 0.5)
