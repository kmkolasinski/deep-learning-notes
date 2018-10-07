"""
This file is copied from GLOW github:
https://github.com/openai/glow/blob/master/tfops.py

And is used only by one class in nets.py: OpenAITemplate
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope, arg_scope

SELU_CONV2D_REG_LOSS = "selu_conv2d_reg_loss"


def default_initial_value(shape, std=0.05):
    return tf.random_normal(shape, 0., std)


def default_initializer(std=0.05):
    return tf.random_normal_initializer(0., std)


def int_shape(x):
    if str(x.get_shape()[0]) != '?':
        return list(map(int, x.get_shape()))
    return [-1] + list(map(int, x.get_shape()[1:]))


# wrapper tf.get_variable, augmented with 'init' functionality
# Get variable with data dependent init


@add_arg_scope
def get_variable_ddi(name, shape, initial_value, dtype=tf.float32, init=False,
                     trainable=True):
    w = tf.get_variable(name, shape, dtype, None, trainable=trainable)
    if init:
        w = w.assign(initial_value)
        with tf.control_dependencies([w]):
            return w
    return w


# Activation normalization
# Convenience function that does centering+scaling

@add_arg_scope
def actnorm(name, x, scale=1., logdet=None, logscale_factor=3.,
            batch_variance=False, reverse=False, init=False, trainable=True):
    if arg_scope([get_variable_ddi], trainable=trainable):
        if not reverse:
            x = actnorm_center(name + "_center", x, reverse)
            x = actnorm_scale(name + "_scale", x, scale, logdet,
                              logscale_factor, batch_variance, reverse, init)
            if logdet != None:
                x, logdet = x
        else:
            x = actnorm_scale(name + "_scale", x, scale, logdet,
                              logscale_factor, batch_variance, reverse, init)
            if logdet != None:
                x, logdet = x
            x = actnorm_center(name + "_center", x, reverse)
        if logdet != None:
            return x, logdet
        return x


# Activation normalization


@add_arg_scope
def actnorm_center(name, x, reverse=False):
    shape = x.get_shape()
    with tf.variable_scope(name):
        assert len(shape) == 2 or len(shape) == 4
        if len(shape) == 2:
            x_mean = tf.reduce_mean(x, [0], keepdims=True)
            b = get_variable_ddi(
                "b", (1, int_shape(x)[1]), initial_value=-x_mean)
        elif len(shape) == 4:
            x_mean = tf.reduce_mean(x, [0, 1, 2], keepdims=True)
            b = get_variable_ddi(
                "b", (1, 1, 1, int_shape(x)[3]), initial_value=-x_mean)

        if not reverse:
            x += b
        else:
            x -= b

        return x


# Activation normalization
@add_arg_scope
def actnorm_scale(name, x, scale=1., logdet=None, logscale_factor=3.,
                  batch_variance=False, reverse=False, init=False,
                  trainable=True):
    shape = x.get_shape()
    with tf.variable_scope(name), arg_scope([get_variable_ddi],
                                            trainable=trainable):
        assert len(shape) == 2 or len(shape) == 4
        if len(shape) == 2:
            x_var = tf.reduce_mean(x ** 2, [0], keepdims=True)
            logdet_factor = 1
            _shape = (1, int_shape(x)[1])

        elif len(shape) == 4:
            x_var = tf.reduce_mean(x ** 2, [0, 1, 2], keepdims=True)
            logdet_factor = int(shape[1]) * int(shape[2])
            _shape = (1, 1, 1, int_shape(x)[3])

        if batch_variance:
            x_var = tf.reduce_mean(x ** 2, keepdims=True)


        logs = get_variable_ddi("logs", _shape, initial_value=tf.log(
            scale / (tf.sqrt(
                x_var) + 1e-6)) / logscale_factor) * logscale_factor
        if not reverse:
            x = x * tf.exp(logs)
        else:
            x = x * tf.exp(-logs)

        if logdet != None:
            dlogdet = tf.reduce_sum(logs) * logdet_factor
            if reverse:
                dlogdet *= -1
            return x, logdet + dlogdet

        return x


# Linear layer with layer norm
@add_arg_scope
def linear(name, x, width, do_weightnorm=True, do_actnorm=True,
           initializer=None, scale=1.):
    initializer = initializer or default_initializer()
    with tf.variable_scope(name):
        n_in = int(x.get_shape()[1])
        w = tf.get_variable("W", [n_in, width],
                            tf.float32, initializer=initializer)
        if do_weightnorm:
            w = tf.nn.l2_normalize(w, [0])
        x = tf.matmul(x, w)
        x += tf.get_variable("b", [1, width],
                             initializer=tf.zeros_initializer())
        if do_actnorm:
            x = actnorm("actnorm", x, scale)
        return x


# Linear layer with zero init
@add_arg_scope
def linear_zeros(name, x, width, logscale_factor=3):
    with tf.variable_scope(name):
        n_in = int(x.get_shape()[1])
        w = tf.get_variable("W", [n_in, width], tf.float32,
                            initializer=tf.zeros_initializer())
        x = tf.matmul(x, w)
        x += tf.get_variable("b", [1, width],
                             initializer=tf.zeros_initializer())
        x *= tf.exp(tf.get_variable("logs",
                                    [1, width],
                                    initializer=tf.zeros_initializer()) * logscale_factor)
        return x


# Slow way to add edge padding
def add_edge_padding(x, filter_size):
    assert filter_size[0] % 2 == 1
    if filter_size[0] == 1 and filter_size[1] == 1:
        return x
    a = (filter_size[0] - 1) // 2  # vertical padding size
    b = (filter_size[1] - 1) // 2  # horizontal padding size
    if True:
        x = tf.pad(x, [[0, 0], [a, a], [b, b], [0, 0]])
        name = "_".join([str(dim) for dim in [a, b, *int_shape(x)[1:3]]])
        pads = tf.get_collection(name)
        if not pads:
            pad = np.zeros([1] + int_shape(x)[1:3] + [1], dtype='float32')
            pad[:, :a, :, 0] = 1.
            pad[:, -a:, :, 0] = 1.
            pad[:, :, :b, 0] = 1.
            pad[:, :, -b:, 0] = 1.
            pad = tf.convert_to_tensor(pad)
            tf.add_to_collection(name, pad)
        else:
            pad = pads[0]
        pad = tf.tile(pad, [tf.shape(x)[0], 1, 1, 1])
        x = tf.concat([x, pad], axis=3)
    else:
        pad = tf.pad(tf.zeros_like(x[:, :, :, :1]) - 1,
                     [[0, 0], [a, a], [b, b], [0, 0]]) + 1
        x = tf.pad(x, [[0, 0], [a, a], [b, b], [0, 0]])
        x = tf.concat([x, pad], axis=3)
    return x


@add_arg_scope
def conv2d(name, x, width, filter_size=[3, 3], stride=[1, 1], pad="SAME",
           do_weightnorm=False, do_actnorm=True, context1d=None, skip=1,
           edge_bias=True):
    with tf.variable_scope(name):
        if edge_bias and pad == "SAME":
            x = add_edge_padding(x, filter_size)
            pad = 'VALID'

        n_in = int(x.get_shape()[3])

        stride_shape = [1] + stride + [1]
        filter_shape = filter_size + [n_in, width]
        w = tf.get_variable("W", filter_shape, tf.float32,
                            initializer=default_initializer())
        if do_weightnorm:
            w = tf.nn.l2_normalize(w, [0, 1, 2])
        if skip == 1:
            x = tf.nn.conv2d(x, w, stride_shape, pad, data_format='NHWC')
        else:
            assert stride[0] == 1 and stride[1] == 1
            x = tf.nn.atrous_conv2d(x, w, skip, pad)
        if do_actnorm:
            x = actnorm("actnorm", x)
        else:
            x += tf.get_variable("b", [1, 1, 1, width],
                                 initializer=tf.zeros_initializer())

        if context1d != None:
            x += tf.reshape(linear("context", context1d,
                                   width), [-1, 1, 1, width])
    return x


@add_arg_scope
def conv2d_zeros(name, x, width, filter_size=[3, 3], stride=[1, 1], pad="SAME",
                 logscale_factor=3, skip=1, edge_bias=True):
    with tf.variable_scope(name):
        if edge_bias and pad == "SAME":
            x = add_edge_padding(x, filter_size)
            pad = 'VALID'

        n_in = int(x.get_shape()[3])
        stride_shape = [1] + stride + [1]
        filter_shape = filter_size + [n_in, width]
        w = tf.get_variable("W", filter_shape, tf.float32,
                            initializer=tf.zeros_initializer())
        if skip == 1:
            x = tf.nn.conv2d(x, w, stride_shape, pad, data_format='NHWC')
        else:
            assert stride[0] == 1 and stride[1] == 1
            x = tf.nn.atrous_conv2d(x, w, skip, pad)
        x += tf.get_variable("b", [1, 1, 1, width],
                             initializer=tf.zeros_initializer())
        x *= tf.exp(tf.get_variable("logs",
                                    [1, width],
                                    initializer=tf.zeros_initializer()) * logscale_factor)
    return x


K = tf.keras.backend
keras = tf.keras


# inspired by loss of VAEs
def fc_selu_reg(x: tf.Tensor, mu: float) -> tf.Tensor:
    # average over filter size
    mean = K.mean(x, axis=0)
    tau_sqr = K.mean(K.square(x), axis=0)
    # average over batch size
    mean_loss = K.mean(K.square(mean))
    tau_loss = K.mean(tau_sqr - K.log(tau_sqr + K.epsilon()))
    return mu * (mean_loss + tau_loss)


def conv2d_selu_regularizer(scale: float):
    def _regularizer_fn(weights: tf.Tensor) -> tf.Tensor:
        shape = K.int_shape(weights)
        num_filters = shape[-1]
        weights = K.reshape(weights, shape=[-1, num_filters])
        with tf.name_scope("SELUConv2DRegLoss"):
            loss = fc_selu_reg(weights, scale)
            tf.add_to_collection(SELU_CONV2D_REG_LOSS, loss)
        return loss

    return _regularizer_fn
