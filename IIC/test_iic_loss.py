import numpy as np
import tensorflow as tf
import iic_loss as iic_loss

tf.compat.v1.enable_eager_execution()
K = tf.keras.backend
keras = tf.keras


class TestComputeJoint(tf.test.TestCase):
    def test_compute_joint(self):

        x_out = tf.constant([[1.0, 0.0], [1.0, 0.0]])
        x_tf_out = tf.constant([[1.0, 0.0], [1.0, 0.0]])

        p_i_j = iic_loss.compute_joint(x_out, x_tf_out)
        self.assertAllClose(p_i_j.numpy(), np.array([[1.0, 0.0], [0.0, 0.0]]))

    def test_compute_joint_v2(self):

        x_out = tf.constant([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        x_tf_out = tf.constant([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        p_i_j = iic_loss.compute_joint(x_out, x_tf_out)
        self.assertAllClose(p_i_j.numpy(), np.array([[2.0 / 3, 0.0], [0.0, 1 / 3.0]]))

    def test_compute_joint_v3(self):

        x_out = tf.constant([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        x_tf_out = tf.constant([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])

        p_i_j = iic_loss.compute_joint(x_out, x_tf_out)
        self.assertAllClose(p_i_j.numpy(), np.ones_like(p_i_j.numpy()) * 0.25)


class TestIICLoss(tf.test.TestCase):
    def test_iic_loss(self):

        x_out = tf.constant([[1.0, 0.0], [1.0, 0.0]])
        x_tf_out = tf.constant([[1.0, 0.0], [1.0, 0.0]])

        loss = iic_loss.iic_loss(x_out, x_tf_out)
        self.assertAllClose(loss.numpy(), 0.0, atol=1e-5)

    def test_iic_loss_v2(self):

        x_out = tf.constant([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        x_tf_out = tf.constant([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])

        loss = iic_loss.iic_loss(x_out, x_tf_out)
        self.assertAllClose(loss, tf.math.log(0.5), atol=1e-4)

    def test_iic_loss_v3(self):

        x_out = tf.constant([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        x_tf_out = tf.constant([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])

        loss = iic_loss.iic_loss(x_out, x_tf_out)
        self.assertAllClose(loss, 0.0)
