"""
A set of Bijectors layers used to build normalizing flow models. This
implementation is very similar to those in the tf.distribution.Bijectors API.
however those layers support the additional flow for the factored out variables
used to model multiscale architectures.
"""
from typing import Optional, Tuple, List, Callable

import numpy as np
import scipy.linalg as salg
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope

K = tf.keras.backend
keras = tf.keras
# x, logdet, z
FlowData = Tuple[tf.Tensor, tf.Tensor, Optional[tf.Tensor]]


def print_shapes(name: str, forward: bool, flow: FlowData) -> None:
    """
    Prints flow data
    Args:
        name: name of the flow class
        forward: whether the flow is in forward mode or not
        flow: flow data
    """
    x_name = "x"
    if not forward:
        x_name = "y"

    if flow[2] is None:
        z_shape = "[None]"
    else:
        z_shape = f"{flow[2].shape.as_list()}"
    print(
        f"{name[:40]:40}: "
        f"{x_name}={flow[0].shape.as_list()}\tz={z_shape}\t"
        f"logdet={flow[1].shape.as_list()}"
    )


def identity_flow(flow: FlowData, forward: bool) -> FlowData:
    """Name flow data nodes. Returns flow passed through the
    tf.identity function."""
    with tf.name_scope("outputs"):
        x, logdet, z = flow
        if forward:
            x = tf.identity(x, name="x")
        else:
            x = tf.identity(x, name="y")
        logdet = tf.identity(logdet, name="lodget")
        if z is not None:
            z = tf.identity(z, name="z")
    return x, logdet, z


class FlowLayer:
    """Abstract flow layer"""
    def __init__(self, name: str = "", **kwargs):
        """
        Initialize flow layer
        Args:
            name: a custom name for the flow layer
            **kwargs: not used
        """
        super(FlowLayer, self).__init__(**kwargs)
        self._forward_inputs = []  # a forward function input tensors
        self._forward_outputs = []  # a forward output tensors
        self._backward_inputs = []
        self._backward_outputs = []
        # This is used only by some layers where the logdet does not
        # depend on the input x.
        self._current_forward_logdet: tf.Tensor = 0.0
        self._current_backward_logdet: tf.Tensor = 0.0
        self._name = name
        self.build()

    def build(self) -> None:
        """Initialize flow variables"""
        pass

    def forward(self, x: tf.Tensor, logdet: tf.Tensor, z: Optional[tf.Tensor], is_training: bool = True) -> FlowData:
        raise NotImplementedError()

    def backward(self, y: tf.Tensor, logdet: tf.Tensor, z: Optional[tf.Tensor], is_training: bool = True) -> FlowData:
        raise NotImplementedError()

    @add_arg_scope
    def __call__(
        self, inputs: FlowData, forward: bool, is_training: bool = True
    ) -> FlowData:
        """Forward or backward flow. If forward = True then this function
        computes:
                flow' = flow.forward(flow)
        otherwise
                flow' = flow.backward(flow)
        This function must satisfy:
                flow == flow.backward(flow.forward(flow))
        Args:
            inputs: input flow data
            forward: whether to use forward mode or not
            is_training: whether the layer is executed in training mode or not.
                This parameter can be used to control behaviour of Dropout
                or BatchNorm layers.

        Returns:
            transformed flow data
        """
        assert isinstance(inputs, tuple)
        scope_name = self._name
        with tf.variable_scope(scope_name), tf.variable_scope(f"{type(self).__name__}"):
            if forward:
                self._forward_inputs.append(inputs)
                outputs = self.forward(*inputs, is_training=is_training)
                outputs = identity_flow(outputs, forward=forward)
                print_shapes(
                    f"{type(self).__name__}/Forward/{scope_name}", True, outputs
                )
                self._forward_outputs.append(outputs)
            else:
                self._backward_inputs.append(inputs)
                outputs = self.backward(*inputs, is_training=is_training)
                outputs = identity_flow(outputs, forward=forward)
                print_shapes(
                    f"{type(self).__name__}/Backward/{scope_name}", False, outputs
                )
                self._backward_outputs.append(outputs)
        return outputs


def InputLayer(x: tf.Tensor) -> FlowData:
    """
    Initialize input flow x is an image
    Args:
        x: input tensor of shape [batch_size, num_channels] or in case of
            images [batch_size, height, width, num_channels]

    Returns:
        x: same as input
        logdet: a zero tensor of shape [batch_size]
        z: a None value. This value is later initialized by FactorOutLayer
    """
    input_shape = K.int_shape(x)
    assert len(input_shape) == 2 or len(input_shape) == 4
    batch_size = input_shape[0]
    assert batch_size is not None
    logdet = tf.zeros([batch_size])
    z = None
    return x, logdet, z


class LogitifyImage(FlowLayer):
    """Apply Tapani Raiko's dequantization and express
    image in terms of logits"""

    def forward(self, x, logdet, z, is_training: bool = True):
        """x should be in range [0, 1]"""
        # corrupt data (Tapani Raiko's dequantization)
        xs = K.int_shape(x)
        assert len(xs) == 4

        y = x * 255.0
        corruption_level = 1.0
        y = y + corruption_level * tf.random_uniform(xs)
        y = y / (255.0 + corruption_level)
        # model logit instead of the x itself
        alpha = 1e-5
        y = y * (1 - alpha) + alpha * 0.5
        new_y = tf.log(y) - tf.log(1 - y)
        dlogdet = tf.reduce_sum(-tf.log(y) - tf.log(1 - y), [1, 2, 3])

        return new_y, logdet + dlogdet, z

    def backward(self, y, logdet, z, is_training: bool = True):
        denominator = 1 + tf.exp(-y)
        x = 1 / denominator
        dlogdet = tf.reduce_sum(-2 * tf.log(denominator) - y, [1, 2, 3])
        return x, logdet + dlogdet, z


class QuantizeImage(FlowLayer):
    """Glow image quantization, according to the paper reducing
    number of bits per image improved model.
    * This flow layer does not change Jacobian, however it is
        approximately invertible.
    * After this Layer the image is mapped from [0, 1] to [-0.5, 0.5] range.
    """
    def __init__(
        self,
        num_bits: int = 8,
        **kwargs
    ):
        """
        Quantize image flow layer. The input image should be normalized to 1.

        Args:
            num_bits: number of bit per color channel
            **kwargs:
        """
        super().__init__(**kwargs)
        self._num_bits = num_bits

    def forward(self, x, logdet, z, is_training: bool = True):
        """x must be in range [0, 1].
        The output image is in range [-0.5, 0.5]"""
        xs = K.int_shape(x)
        assert len(xs) == 4

        num_bins = 2. ** self._num_bits

        if self._num_bits < 8:
            x = tf.floor(255 * x / 2 ** (8 - self._num_bits))
            x = x / num_bins
        x = x - 0.5
        y = x + tf.random_uniform(tf.shape(x), 0, 1. / num_bins)
        return y, logdet, z

    def backward(self, y, logdet, z, is_training: bool = True):
        x = y + .5
        return x, logdet, z

    def to_uint8(self, y: tf.Tensor) -> tf.Tensor:
        """
        Convert images to uint8 type.
        Args:
            y: tensor of shape [batch_size, height, width, num_channels],
               being the output of the backward() function

        Returns:
            x: tensor of the same shape
        """
        num_bins = 2. ** self._num_bits
        x = tf.cast(tf.clip_by_value(
            tf.floor(y * num_bins) * (256. / num_bins), 0, 255
        ), 'uint8')
        return x


class SqueezingLayer(FlowLayer):
    """Squeeze image spatial dimensionality. For example is the
    input image has:
        shape(x) = [batch_size, size, size, num_channels],
    this function will return new tensor x' with shape:
        shape(x') = [batch_size, size//2, size//2, num_channels * 2]

    Important notes:
        * size must be divisible by 2
        * if z is not None it will be squeezed in the same manner
        * this is volume preserving operation, hence logdet is unchanged
    """

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
    directly to the latent space. This layer can be used to model the
    multi-scale architecture.

    Important notes:
        * if z is not None it will be concatenated with factored out tensor
        * this is volume preserving operation, hence logdet is unchanged

    """

    def __init__(self, name: str = "", **kwargs):
        super().__init__(name=name, **kwargs)
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
    """Chain multiple layers into single flow"""

    def __init__(self, layers: List[FlowLayer], name: str = "", **kwargs):
        """Initialize ChainLayer

        Args:
            layers: a list of flow layers
            name: a custom name of the layer
            **kwargs:
        """
        super().__init__(name=name, **kwargs)
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


class _ActnormBaseLayer(FlowLayer):
    """An implementation of the ActNormLayer, this base class serves several
    utility functions used by their derivatives: Bias and Scale layer"""
    def __init__(
        self,
        name: str = "",
        input_shape: Optional[Tuple[int, int, int, int]] = None,
        **kwargs,
    ) -> None:
        """
        Args:
            name: a custom name of the layer
            input_shape: an optional expected input tensor shape
                [batch_size, height, width, num_channels] in case of images
                or [batch_size, num_channels] when using dense layers. If not provided
                the layer will be initialized when calling forward function.
            **kwargs:
        """
        # input_shape use it whenever you know the input shape
        if input_shape is not None:
            # do some checks
            assert len(input_shape) == 2 or len(input_shape) == 4
        else:
            self._input_shape = input_shape
        super().__init__(name=name, **kwargs)

    @property
    def variable_shape(self) -> Tuple[int, ...]:
        """A shape of the bias or scale variable"""
        shape = None
        if len(self._input_shape) == 2:
            shape = (1, self._input_shape[1])
        elif len(self._input_shape) == 4:
            shape = (1, 1, 1, self._input_shape[3])
        return shape

    @property
    def forward_output_moments(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """Estimates mean and variance variables of the actnorm output
        tensors i.e this function computes:
                y = actnorm.forward(x)
                mean = mean(y)
                var = mean(|y - mean(y)|**2)
        Note, that those statistics are used to implement Data Dependent
        Initialization.

        Returns:
            mean and variance of the output activations of the actnorm layer of
                shape [1, 1, 1, num_channels] in case of images and [1, num_channels]
                in case of dense layers.
        """
        assert len(self._forward_outputs) == 1
        x, logdet, z = self._forward_outputs[0]

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
        name: str = "",
        input_shape: Optional[Tuple[int, int, int, int]] = None,
        **kwargs,
    ) -> None:
        """
        A bias bijector, it computes:
                y = x + bias
        Since this function is a volume preserving operation it does not change
        the Jacobian.
        """
        super().__init__(name=name, input_shape=input_shape, **kwargs)
        self._bias_t: tf.Variable = None

    def get_ddi_init_ops(self, num_init_iterations: int = 0) -> tf.Operation:
        """Return tensorflow update operation which can be used to
        initialize bias variable, such that the output activations have
        unit variance and mean zero. This function can be called only after
        forward flow graph construction i.e.

                output_flow = actnorm(input_flow)
                init_op = actnorm.get_ddi_init_ops()
                sess.run(init_op)

        Args:
            num_init_iterations: the number of iteration used to estimate the
                variables based on the output activation statistics. Zero value
                corresponds to the initialization proposed in the Glow paper.
                Setting larger values allows to used more batches to iteratively
                initialize layer variables.

        Returns:
            init_op: a tensorflow Operation
        """
        x_mean, _ = self.forward_output_moments
        # initialize bias
        n = num_init_iterations
        omega = tf.to_float(np.exp(-min(1, n) / max(1, n)))
        new_bias = self._bias_t * (1 - omega) - omega * x_mean
        bias_assign_op = tf.assign_add(self._bias_t, new_bias, name="bias_assign")

        return bias_assign_op

    def build(self):
        if self._input_shape is None:
            return

        self._bias_t = tf.get_variable(
            "bias", self.variable_shape, tf.float32, initializer=tf.zeros_initializer()
        )

    def forward(self, x, logdet, z, is_training: bool = True) -> FlowData:
        if self._input_shape is None:
            self._input_shape = K.int_shape(x)
            self.build()

        y = x + self._bias_t
        return y, logdet, z

    def backward(self, y, logdet, z, is_training: bool = True) -> FlowData:
        x = y - self._bias_t
        return x, logdet, z


class ActnormScaleLayer(_ActnormBaseLayer):
    def __init__(
        self,
        name: str = "",
        input_shape: Optional[Tuple[int, int, int, int]] = None,
        scale: float = 1.0,
        **kwargs,
    ):
        """
        An affine scale bijector, it computes:
                    y = scale * x

        Args:
            scale: a scale of the variance of the initial value of the
                log_scale parameter when using data dependent initialization.
                See get_ddi_init_ops for the implementation.
        """
        super().__init__(name=name, input_shape=input_shape, **kwargs)

        self._scale = scale
        self._log_scale_t: tf.Variable = None

    def get_ddi_init_ops(self, num_init_iterations: int = 0) -> tf.Operation:
        """Return tensorflow update operation which can be used to
        initialize bias variable, such that the output activations have
        unit variance and mean zero. This function can be called only after
        forward flow graph construction i.e.

                output_flow = actnorm(input_flow)
                init_op = actnorm.get_ddi_init_ops()
                sess.run(init_op)

        Args:
            num_init_iterations: the number of iteration used to estimate the
                variables based on the output activation statistics. Zero value
                corresponds to the initialization proposed in the Glow paper.
                Setting larger values allows to used more batches to iteratively
                initialize layer variables.

        Returns:
            init_op: a tensorflow Operation
        """
        _, x_var = self.forward_output_moments

        # derived for iterative initialization
        var = self._log_scale_t + tf.log(self._scale / (tf.sqrt(x_var) + 1e-6))

        n = num_init_iterations
        omega = tf.to_float(np.exp(-min(1, n) / max(1, n)))
        new_scale = self._log_scale_t * (1 - omega) + omega * var

        log_scale_assign_op = tf.assign(
            self._log_scale_t, new_scale, name="log_scale_assign"
        )
        return log_scale_assign_op

    def build(self):
        if self._input_shape is None:
            return
        self._log_scale_t = tf.get_variable(
            "log_scale",
            self.variable_shape,
            tf.float32,
            initializer=tf.zeros_initializer(),
        )

    def forward(self, x, logdet, z, is_training: bool = True) -> FlowData:
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

    def backward(self, y, logdet, z, is_training: bool = True) -> FlowData:

        x = y * tf.exp(-self._log_scale_t)
        self._current_backward_logdet = -self._current_forward_logdet
        dlogdet = self._current_backward_logdet
        return x, logdet + dlogdet, z


class ActnormLayer(FlowLayer):
    def __init__(
        self,
        name: str = "",
        input_shape: Optional[Tuple[int, int, int, int]] = None,
        scale: float = 1.0,
        **kwargs,
    ):
        """
        An implementation of the actnorm layer:
                y = w * x + bias

        Args:
            scale: a scale parameter of the variance of the initial value of the
                log_scale parameter when using data dependent initialization.
                See get_ddi_init_ops for the implementation.
        """
        super().__init__(name=name, **kwargs)
        self._bias_layer = ActnormBiasLayer(input_shape=input_shape)
        self._scale_layer = ActnormScaleLayer(input_shape=input_shape, scale=scale)
        self._chain = ChainLayer([self._bias_layer, self._scale_layer])

    def get_ddi_init_ops(self, num_init_iterations: int = 0) -> tf.Operation:
        bias_update_op = self._bias_layer.get_ddi_init_ops(num_init_iterations)
        with tf.control_dependencies([bias_update_op]):
            scale_update_op = self._scale_layer.get_ddi_init_ops(num_init_iterations)
            update_ops = tf.group([bias_update_op, scale_update_op])
        return update_ops

    def forward(self, x, logdet, z, is_training: bool = True) -> FlowData:
        return self._chain((x, logdet, z), forward=True, is_training=is_training)

    def backward(self, y, logdet, z, is_training: bool = True)-> FlowData:
        return self._chain((y, logdet, z), forward=False, is_training=is_training)


class InvertibleConv1x1Layer(FlowLayer):
    def __init__(self, name: str = "", use_lu_decomposition: bool = True, **kwargs):
        """
        Reimplementation of the Invertible 1D convolution see Glow paper.
        This code has been copied from the glow source code and changed to fit
        my API.

        Args:
            name: custom name for the layer
            use_lu_decomposition: compute inverse of the kernel matrix using LU
                decomposition. If False use naive approach.
            **kwargs:
        """
        self._use_lu_decomposition = use_lu_decomposition
        self._input_shape = None
        # only trainable weights
        self._weights: List[tf.Variable] = []
        # only non-trainable weights
        self._non_trainable_weights: List[tf.Variable] = []
        self._kernel_t: tf.Tensor = None
        self._inv_kernel_t: tf.Tensor = None
        self._dlogdet_t: tf.Tensor = None
        super().__init__(name=name, **kwargs)

    def build(self):
        if self._input_shape is None:
            return
        assert len(self._input_shape) == 4
        dtype = "float64"
        shape = self._input_shape
        num_channels = shape[3]
        w_shape = [num_channels, num_channels]
        kernel_shape = [1, 1, num_channels, num_channels]
        # Sample a random orthogonal matrix
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype("float32")

        if not self._use_lu_decomposition:

            w = tf.get_variable("kernel", dtype=tf.float32, initializer=w_init)
            dlogdet = (
                tf.cast(
                    tf.log(tf.abs(tf.matrix_determinant(tf.cast(w, dtype)))), "float32"
                )
                * shape[1]
                * shape[2]
            )

            self._weights = [w]
            self._dlogdet_t = dlogdet
            self._kernel_t = tf.reshape(w, kernel_shape)
            self._inv_kernel_t = tf.reshape(tf.matrix_inverse(w), kernel_shape)

        else:
            np_p, np_l, np_u = salg.lu(w_init, permute_l=False)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(abs(np_s))
            np_u = np.triu(np_u, k=1)

            p_mat = tf.get_variable("P_mat", initializer=np_p, trainable=False)
            l_mat = tf.get_variable("L_mat", initializer=np_l)
            sign_s = tf.get_variable("sign_S", initializer=np_sign_s, trainable=False)
            log_s = tf.get_variable("log_S", initializer=np_log_s)
            u_mat = tf.get_variable("U_mat", initializer=np_u)

            self._weights = [l_mat, log_s, u_mat]
            self._non_trainable_weights = [p_mat, sign_s]

            p_mat = tf.cast(p_mat, dtype)
            l_mat = tf.cast(l_mat, dtype)
            sign_s = tf.cast(sign_s, dtype)
            log_s = tf.cast(log_s, dtype)
            u_mat = tf.cast(u_mat, dtype)

            l_mask = np.tril(np.ones(w_shape, dtype=dtype), -1)
            l_mat = l_mat * l_mask + tf.eye(*w_shape, dtype=dtype)
            u_mat = u_mat * np.transpose(l_mask) + tf.diag(sign_s * tf.exp(log_s))
            w = tf.matmul(p_mat, tf.matmul(l_mat, u_mat))

            # inverse w
            u_inv = tf.matrix_inverse(u_mat)
            l_inv = tf.matrix_inverse(l_mat)
            p_inv = tf.matrix_inverse(p_mat)
            w_inv = tf.matmul(u_inv, tf.matmul(l_inv, p_inv))

            w = tf.cast(w, tf.float32)
            w_inv = tf.cast(w_inv, tf.float32)
            log_s = tf.cast(log_s, tf.float32)
            self._dlogdet_t = tf.reduce_sum(log_s) * shape[1] * shape[2]
            self._kernel_t = tf.reshape(w, kernel_shape)
            self._inv_kernel_t = tf.reshape(w_inv, kernel_shape)

    def forward(self, x, logdet, z, is_training: bool = True) -> FlowData:
        if self._input_shape is None:
            self._input_shape = K.int_shape(x)
            self.build()

        y = tf.nn.conv2d(x, self._kernel_t, [1, 1, 1, 1], "SAME", data_format="NHWC")
        return y, logdet + self._dlogdet_t, z

    def backward(self, y, logdet, z, is_training: bool = True) -> FlowData:

        x = tf.nn.conv2d(
            y, self._inv_kernel_t, [1, 1, 1, 1], "SAME", data_format="NHWC"
        )
        return x, logdet - self._dlogdet_t, z


class AffineCouplingLayer(FlowLayer):
    def __init__(
        self,
        shift_and_log_scale_fn: Callable[[tf.Tensor], tf.Tensor],
        name: str = "",
        log_scale_fn: Callable[[tf.Tensor], tf.Tensor] = lambda x: tf.exp(tf.clip_by_value(x, -15.0, 15.0)),
        **kwargs,
    ):
        """
        Implementation of the affine coupling layer (see RealNVP or NICE) paper
        for more details.
        Args:
            shift_and_log_scale_fn: a function which takes for the input tensor
                of shape [batch_size, width, height, num_channels] and return
                shift and log_scale tensors of the same shape.
            name: a custom name of the flow
            log_scale_fn: a log scale function
            **kwargs:
        """
        super().__init__(name=name, **kwargs)
        self._shift_and_log_scale_fn = shift_and_log_scale_fn
        self._log_scale_fn = log_scale_fn

    def forward(self, x, logdet, z, is_training: bool = True) -> FlowData:
        input_shape = K.int_shape(x)
        assert len(input_shape) == 4
        num_channels = input_shape[3]
        # split tensor along channel axis
        x1 = x[:, :, :, : num_channels // 2]
        x2 = x[:, :, :, num_channels // 2 :]

        shift, log_scale = self._shift_and_log_scale_fn(x1)

        if shift is not None:
            x2 += shift

        if log_scale is not None:
            scale = self._log_scale_fn(log_scale)
            x2 *= scale
            dlogdet = tf.reduce_sum(tf.log(scale), axis=[1, 2, 3])
        else:
            # coupling layer is volume preserving when scale is 1.0
            dlogdet = 0.0

        y = tf.concat([x1, x2], axis=3)
        return y, logdet + dlogdet, z

    def backward(self, y, logdet, z, is_training: bool = True) -> FlowData:
        input_shape = K.int_shape(y)
        assert len(input_shape) == 4
        num_channels = input_shape[3]

        y1 = y[:, :, :, : num_channels // 2]
        y2 = y[:, :, :, num_channels // 2 :]

        shift, log_scale = self._shift_and_log_scale_fn(y1)
        if log_scale is not None:
            scale = self._log_scale_fn(log_scale)
            dlogdet = -tf.reduce_sum(tf.log(scale), axis=[1, 2, 3])
            y2 /= scale
        else:
            dlogdet = 0.0
        if shift is not None:
            y2 -= shift

        x = tf.concat([y1, y2], axis=3)
        return x, logdet + dlogdet, z
