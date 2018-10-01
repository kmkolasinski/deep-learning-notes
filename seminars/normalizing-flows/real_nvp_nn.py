from typing import Optional, Tuple, List

import numpy as np
import tensorflow as tf
K = tf.keras.backend
keras = tf.keras
from tensorflow.contrib.framework.python.ops import add_arg_scope, arg_scope
from tensorflow.contrib.layers import variance_scaling_initializer


def int_shape(x):
    return list(map(int, x.get_shape()))


# Abstract class that can propagate both forward/backward,
# along with jacobians.
class Layer:
    def forward_and_jacobian(self, x, sum_log_det_jacobians, z):
        raise NotImplementedError(str(type(self)))

    def backward(self, y, z):
        raise NotImplementedError(str(type(self)))


# The coupling layer.
# Contains code for both checkerboard and channelwise masking.
class CouplingLayer(Layer):

    # |mask_type| can be 'checkerboard0', 'checkerboard1', 'channel0', 'channel1'
    def __init__(self, mask_type, name="Coupling"):
        self.mask_type = mask_type
        self.name = name

    # Batch normalization.
    # TODO: Moving average batch normaliation
    def batch_norm(self, x):
        mu = tf.reduce_mean(x)
        sig2 = tf.reduce_mean(tf.square(x - mu))
        x = (x - mu) / tf.sqrt(sig2 + 1.0e-6)
        return x, sig2

    # Weight normalization technique
    def get_normalized_weights(self, name, weights_shape):
        weights = tf.get_variable(
            name, weights_shape, tf.float32, tf.contrib.layers.xavier_initializer()
        )
        scale = tf.get_variable(
            name + "_scale",
            [1],
            tf.float32,
            tf.contrib.layers.xavier_initializer(),
            regularizer=tf.contrib.layers.l2_regularizer(5e-5),
        )
        norm = tf.sqrt(tf.reduce_sum(tf.square(weights)))
        return weights / norm * scale

    # corresponds to the function m and l in the RealNVP paper
    # (Function m and l became s and t in the new version of the paper)
    def function_l_m(self, x, mask, name="function_l_m"):
        with tf.variable_scope(name):
            channel = 64
            padding = "SAME"
            xs = int_shape(x)
            kernel_h = 3
            kernel_w = 3
            input_channel = xs[3]
            y = x

            y, _ = self.batch_norm(y)
            weights_shape = [1, 1, input_channel, channel]
            weights = self.get_normalized_weights("weights_input", weights_shape)

            y = tf.nn.conv2d(y, weights, [1, 1, 1, 1], padding=padding)
            y, _ = self.batch_norm(y)
            y = tf.nn.relu(y)

            skip = y
            # Residual blocks
            num_residual_blocks = 8
            for r in range(num_residual_blocks):
                weights_shape = [kernel_h, kernel_w, channel, channel]
                weights = self.get_normalized_weights("weights%d_1" % r, weights_shape)
                y = tf.nn.conv2d(y, weights, [1, 1, 1, 1], padding=padding)
                y, _ = self.batch_norm(y)
                y = tf.nn.relu(y)
                weights_shape = [kernel_h, kernel_w, channel, channel]
                weights = self.get_normalized_weights("weights%d_2" % r, weights_shape)
                y = tf.nn.conv2d(y, weights, [1, 1, 1, 1], padding=padding)
                y, _ = self.batch_norm(y)
                y += skip
                y = tf.nn.relu(y)
                skip = y

            # 1x1 convolution for reducing dimension
            weights = self.get_normalized_weights(
                "weights_output", [1, 1, channel, input_channel * 2]
            )
            y = tf.nn.conv2d(y, weights, [1, 1, 1, 1], padding=padding)

            # For numerical stability, apply tanh and then scale
            y = tf.tanh(y)
            scale_factor = self.get_normalized_weights("weights_tanh_scale", [1])
            y *= scale_factor

            # The first half defines the l function
            # The second half defines the m function
            l = y[:, :, :, :input_channel] * (-mask + 1)
            m = y[:, :, :, input_channel:] * (-mask + 1)

            return l, m

    # returns constant tensor of masks
    # |xs| is the size of tensor
    # |mask_type| can be 'checkerboard0', 'checkerboard1', 'channel0', 'channel1'
    # |b| has the dimension of |xs|
    def get_mask(self, xs, mask_type):

        if "checkerboard" in mask_type:
            unit0 = tf.constant([[0.0, 1.0], [1.0, 0.0]])
            unit1 = -unit0 + 1.0
            unit = unit0 if mask_type == "checkerboard0" else unit1
            unit = tf.reshape(unit, [1, 2, 2, 1])
            b = tf.tile(unit, [xs[0], xs[1] // 2, xs[2] // 2, xs[3]])
        elif "channel" in mask_type:
            white = tf.ones([xs[0], xs[1], xs[2], xs[3] // 2])
            black = tf.zeros([xs[0], xs[1], xs[2], xs[3] // 2])
            if mask_type == "channel0":
                b = tf.concat(3, [white, black])
            else:
                b = tf.concat(3, [black, white])

        bs = int_shape(b)
        assert bs == xs

        return b

    # corresponds to the coupling layer of the RealNVP paper
    # |mask_type| can be 'checkerboard0', 'checkerboard1', 'channel0', 'channel1'
    # log_det_jacobian is a 1D tensor of size (batch_size)
    def forward_and_jacobian(self, x, sum_log_det_jacobians, z):
        with tf.variable_scope(self.name):
            xs = int_shape(x)
            b = self.get_mask(xs, self.mask_type)

            # masked half of x
            x1 = x * b
            l, m = self.function_l_m(x1, b)
            y = x1 + tf.mul(
                -b + 1.0, x * tf.check_numerics(tf.exp(l), "exp has NaN") + m
            )
            log_det_jacobian = tf.reduce_sum(l, [1, 2, 3])
            sum_log_det_jacobians += log_det_jacobian

            return y, sum_log_det_jacobians, z

    def backward(self, y, z):
        with tf.variable_scope(self.name, reuse=True):
            ys = int_shape(y)
            b = self.get_mask(ys, self.mask_type)

            y1 = y * b
            l, m = self.function_l_m(y1, b)
            x = y1 + tf.mul(y * (-b + 1.0) - m, tf.exp(-l))
            return x, z


# Given the output of the network and all jacobians,
# compute the log probability.
# Equation (3) of the RealNVP paper
def compute_log_prob_x(z, sum_log_det_jacobians):
    # y is assumed to be in standard normal distribution
    # 1/sqrt(2*pi)*exp(-0.5*x^2)
    zs = int_shape(z)
    K = zs[1] * zs[2] * zs[3]  # dimension of the Gaussian distribution

    log_density_z = -0.5 * tf.reduce_sum(tf.square(z), [1, 2, 3]) - 0.5 * K * np.log(
        2 * np.pi
    )

    log_density_x = log_density_z + sum_log_det_jacobians

    # to go from density to probability, one can
    # multiply the density by the width of the
    # discrete probability area, which is 1/256.0, per dimension.
    # The calculation is performed in the log space.
    log_prob_x = log_density_x - K * tf.log(256.0)

    return log_prob_x


# Computes the loss of the network.
# It is chosen so that the probability P(x) of the
# natural images is maximized.
def loss(z, sum_log_det_jacobians):
    return -tf.reduce_sum(compute_log_prob_x(z, sum_log_det_jacobians))


# x, logdet, z
FlowData = Tuple[tf.Tensor, tf.Tensor, Optional[tf.Tensor]]


def print_shapes(name: str, forward: bool, flow: FlowData):
    x_name = "x"
    if not forward:
        x_name = "y"

    if flow[2] is None:
        z_shape = "[None]"
    else:
        z_shape = f"{flow[2].shape.as_list()}"
    print(f"{name} "
          f"{x_name}={flow[0].shape.as_list()} "
          f"logdet={flow[1].shape.as_list()} z={z_shape}")


class FlowLayer:
    def __init__(self, **kwargs):
        super(FlowLayer, self).__init__(**kwargs)
        self._forward_inputs = []
        self._forward_outputs = []
        self._backward_inputs = []
        self._backward_outputs = []
        self._current_forward_logdet: tf.Tensor = 0.0
        self._current_backward_logdet: tf.Tensor = 0.0
        self.build()

    def build(self):
        pass

    def forward(self, x, logdet, z, is_training: bool = True) -> FlowData:
        raise NotImplementedError()

    def backward(self, y, logdet, z, is_training: bool = True) -> FlowData:
        raise NotImplementedError()

    @add_arg_scope
    def __call__(self, inputs: FlowData, forward: bool, is_training: bool = True) -> FlowData:
        assert isinstance(inputs, tuple)
        if forward:
            with tf.name_scope(f"{type(self).__name__}Forward"):
                self._forward_inputs.append(inputs)
                outputs = self.forward(*inputs, is_training=is_training)
                print_shapes(f"{type(self).__name__}Forward", True, outputs)
                self._forward_outputs.append(outputs)
            return outputs
        else:
            with tf.name_scope(f"{type(self).__name__}Backward"):
                self._backward_inputs.append(inputs)
                outputs = self.backward(*inputs, is_training=is_training)
                print_shapes(f"{type(self).__name__}Backward", False, outputs)
                self._backward_outputs.append(outputs)
            return outputs


def InputLayer(x: tf.Tensor) -> FlowData:
    """Initialize input flow"""
    input_shape = K.int_shape(x)
    assert len(input_shape) == 2 or len(input_shape) == 4

    batch_size = input_shape[0]

    assert batch_size is not None

    logdet = tf.zeros([batch_size])
    z = None
    return x, logdet, z


class SqueezingLayer(FlowLayer):
    """Change dimensionality of the layer """
    def forward(self, x, logdet, z, is_training: bool = True):

        xs = K.int_shape(x)
        assert len(xs) == 4
        assert xs[1] % 2 == 0 and xs[2] % 2 == 0
        y = tf.space_to_depth(x, 2)
        if z is not None:
            z = tf.space_to_depth(z, 2)

        return y, logdet, z

    def backward(self, y, logdet, z, is_training: bool = True):
        ys = K.int_shape(y)
        assert len(ys) == 4
        assert ys[3] % 4 == 0
        x = tf.depth_to_space(y, 2)
        if z is not None:
            z = tf.depth_to_space(z, 2)

        return x, logdet, z


class FactorOutLayer(FlowLayer):
    """
    The layer that factors out half of the variables
    directly to the latent space.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.split_size = None

    def forward(self, x, logdet, z, is_training: bool = True):

        xs = K.int_shape(x)
        assert len(xs) == 4
        split = xs[3] // 2
        self.split_size = split

        new_z = x[:, :, :, :split]
        x = x[:, :, :, split:]

        if z is not None:
            z = tf.concat([z, new_z], axis=3)
        else:
            z = new_z

        return x, logdet, z

    def backward(self, y, logdet, z, is_training: bool = True):
        assert self.split_size is not None

        if y is None:
            split = self.split_size
        else:
            split = K.int_shape(y)[3]

        assert len(K.int_shape(y)) == 4
        new_y = z[:, :, :, -split:]
        z = z[:, :, :, :-split]

        assert K.int_shape(new_y)[3] == split
        if y is not None:
            x = tf.concat([new_y, y], axis=3)
        else:
            x = new_y
        return x, logdet, z


class ChainLayer(FlowLayer):
    """chain multiple flows"""

    def __init__(self, layers: List[FlowLayer], **kwargs):
        super().__init__(**kwargs)
        self._layers = layers

    def forward(self, x, logdet, z, is_training: bool = True):
        flow = x, logdet, z
        for layer in self._layers:
            flow = layer(flow, forward=True, is_training=is_training)
        return flow

    def backward(self, y, logdet, z, is_training: bool = True):
        flow = y, logdet, z
        for layer in reversed(self._layers):
            flow = layer(flow, forward=False, is_training=is_training)
        return flow


@add_arg_scope
def get_variable_ddi(name, shape, initial_value, dtype=tf.float32, init=False, trainable=True):
    w = tf.get_variable(name, shape, dtype, None, trainable=trainable)
    if init:
        w = w.assign(initial_value)
        with tf.control_dependencies([w]):
            return w
    return w


@add_arg_scope
def actnorm(name, x, scale=1., logdet=None, logscale_factor=3., batch_variance=False, reverse=False, init=False, trainable=True):
    if arg_scope([get_variable_ddi], trainable=trainable):
        if not reverse:
            x = actnorm_center(name+"_center", x, reverse)
            x = actnorm_scale(name+"_scale", x, scale, logdet,
                              logscale_factor, batch_variance, reverse, init)
            if logdet != None:
                x, logdet = x
        else:
            x = actnorm_scale(name + "_scale", x, scale, logdet,
                              logscale_factor, batch_variance, reverse, init)
            if logdet != None:
                x, logdet = x
            x = actnorm_center(name+"_center", x, reverse)
        if logdet != None:
            return x, logdet
        return x


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


@add_arg_scope
def actnorm_scale(name, x, scale=1., logdet=None, logscale_factor=3., batch_variance=False, reverse=False, init=False, trainable=True):
    shape = x.get_shape()
    with tf.variable_scope(name), arg_scope([get_variable_ddi], trainable=trainable):
        assert len(shape) == 2 or len(shape) == 4
        if len(shape) == 2:
            x_var = tf.reduce_mean(x**2, [0], keepdims=True)
            logdet_factor = 1
            _shape = (1, int_shape(x)[1])

        elif len(shape) == 4:
            x_var = tf.reduce_mean(x**2, [0, 1, 2], keepdims=True)
            logdet_factor = int(shape[1])*int(shape[2])
            _shape = (1, 1, 1, int_shape(x)[3])

        if batch_variance:
            x_var = tf.reduce_mean(x**2, keepdims=True)

        logs = get_variable_ddi("logs", _shape, initial_value=tf.log(
            scale/(tf.sqrt(x_var)+1e-6)) / logscale_factor) * logscale_factor
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


class _ActnormBaseLayer(FlowLayer):
    def __init__(
            self,
            input_shape: Optional[Tuple[int, int, int, int]] = None,
            **kwargs
    ):
        # input_shape use it whenever you know the input shape
        if input_shape is not None:
            # do some checks
            assert len(input_shape) == 2 or len(input_shape) == 4
        else:
            self._input_shape = input_shape

        super().__init__(**kwargs)

    @property
    def variable_shape(self) -> Tuple[int, ...]:
        shape = None
        if len(self._input_shape) == 2:
            shape = (1, self._input_shape[1])
        elif len(self._input_shape) == 4:
            shape = (1, 1, 1, self._input_shape[3])
        return shape

    @property
    def forward_output_moments(self) -> Tuple[tf.Tensor, tf.Tensor]:
        assert len(self._forward_outputs) == 1
        x, logdet, z = self._forward_outputs[0]
        x_mean = None
        x_var = None

        if len(self._input_shape) == 2:
            x_mean = tf.reduce_mean(x, [0], keepdims=True)
            x_var = tf.reduce_mean((x - x_mean) ** 2, [0], keepdims=True)

        elif len(self._input_shape) == 4:
            x_mean = tf.reduce_mean(x, [0, 1, 2], keepdims=True)
            x_var = tf.reduce_mean((x - x_mean) ** 2, [0, 1, 2], keepdims=True)
        else:
            raise ValueError("Unknown shape")

        return x_mean, x_var


class ActnormBiasLayer(_ActnormBaseLayer):
    def __init__(
            self,
            input_shape: Optional[Tuple[int, int, int, int]] = None,
            **kwargs
    ):
        super().__init__(input_shape=input_shape, **kwargs)
        self._bias_t: tf.Variable = None

    def get_ddi_init_ops(self, num_init_iterations: int = 0):
        x_mean, _ = self.forward_output_moments
        # initialize bias
        n = num_init_iterations
        omega = tf.to_float(np.exp(-min(1, n)/max(1, n)))
        new_bias = self._bias_t * (1 - omega) - omega * x_mean
        bias_assign_op = tf.assign_add(self._bias_t, new_bias, name='bias_assign')

        return bias_assign_op

    def build(self):
        if self._input_shape is None:
            return

        self._bias_t = tf.get_variable(
            "bias", self.variable_shape, tf.float32, initializer=tf.zeros_initializer()
        )

    def forward(self, x, logdet, z, is_training: bool = True):
        if self._input_shape is None:
            self._input_shape = K.int_shape(x)
            self.build()

        y = x + self._bias_t
        return y, logdet, z

    def backward(self, y, logdet, z, is_training: bool = True):
        x = y - self._bias_t
        return x, logdet, z


class ActnormScaleLayer(_ActnormBaseLayer):
    def __init__(
            self,
            input_shape: Optional[Tuple[int, int, int, int]] = None,
            scale: float = 1.0,
            **kwargs
    ):
        super().__init__(input_shape=input_shape, **kwargs)

        self._scale = scale
        self._log_scale_t: tf.Variable = None

    def get_ddi_init_ops(self, num_init_iterations: int = 0):
        _, x_var = self.forward_output_moments

        # derived for iterative initialization
        var = self._log_scale_t + tf.log(self._scale / (tf.sqrt(x_var) + 1e-6))

        n = num_init_iterations
        omega = tf.to_float(np.exp(-min(1, n) / max(1, n)))
        new_scale = self._log_scale_t * (1 - omega) + omega * var

        log_scale_assign_op = tf.assign(
            self._log_scale_t, new_scale, name='log_scale_assign'
        )
        return log_scale_assign_op

    def build(self):
        if self._input_shape is None:
            return
        self._log_scale_t = tf.get_variable(
            "log_scale", self.variable_shape, tf.float32, initializer=tf.zeros_initializer())

    def forward(self, x, logdet, z, is_training: bool = True):
        if self._input_shape is None:
            self._input_shape = K.int_shape(x)
            self.build()

        y = x * tf.exp(self._log_scale_t)

        logdet_factor = 1
        if len(self._input_shape) == 2:
            logdet_factor = 1
        elif len(self._input_shape) == 4:
            logdet_factor = self._input_shape[1] * self._input_shape[2]

        dlogdet = logdet_factor * tf.reduce_sum(self._log_scale_t)
        self._current_forward_logdet = dlogdet

        return y, logdet + dlogdet, z

    def backward(self, y, logdet, z, is_training: bool = True):

        x = y * tf.exp(- self._log_scale_t)
        self._current_backward_logdet = - self._current_forward_logdet
        dlogdet = self._current_backward_logdet
        return x, logdet + dlogdet, z