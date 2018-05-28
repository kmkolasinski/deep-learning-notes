import tensorflow as tf
from typing import Union, Optional

_EPSILON = 1e-6


class NormalizedSGD(tf.train.Optimizer):
    """An implementation of normalized Stochastic Gradient Descent.

    @@__init__
    """
    def __init__(
            self,
            lr: Union[float, tf.Tensor],
            lr_update: float = 0.01,
            lr_max: Optional[float] = 0.01,
            lr_min: Optional[float] = 1e-5,
            lr_force: float = 0.0,
            norm: str = 'max',
            noise_amplitude: Union[float, tf.Tensor]=0.01,
            use_locking: bool=False,
            name: str="NormalizedSGD"
    ) -> None:
        """Optimizer constructor.

        Args:
            lr: learning rate value or tensor
            noise_amplitude: the amplitude of the noise to be added
                to the normalized gradients. This should help to escape
                flat regions. Noise is added with amplitude which is
                proportional to the learning rate i.e.
                         final_amplitude = noise_amplitude * learning_rate
                Noise is sampled from uniform distribution U[-1, 1].
            lr_update: relative learning rate update step. New lr is computed
                approximately as lr' = lr * (1 - lr_update * cos(a)), a is the
                angle between gradient in the k-th step and the k-1 step.
            lr_max: max value of lr, if None then lr_max=lr
            lr_min: min value of lr, if None then lr_min=0.0
            lr_force: relative force lr to increase in consecutive steps.
            use_locking: Bool. If True apply use locks to prevent concurrent
                updates to variables.
            name: A non-empty string. The name to use for accumulators created
                for the optimizer.
        """
        super(NormalizedSGD, self).__init__(use_locking, name)

        if norm not in ['max', 'l2']:
            raise ValueError(f'Unexpected norm type `{norm}`.')

        self._lr = lr
        self._lr_update = lr_update
        self._lr_max = lr_max if lr_max is not None else lr
        self._lr_min = lr_min if lr_min is not None else 0.0
        self._lr_force = lr_force
        self._norm = norm
        self._alpha = noise_amplitude

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._alpha_t = None
        self._lr_update_t = None
        self._lr_max_t = None
        self._lr_min_t = None
        self._lr_force_t = None
        self._lr_variables = []

    def _prepare(self):
        self._lr_t = tf.convert_to_tensor(self._lr, name="lr")
        self._alpha_t = tf.convert_to_tensor(self._alpha, name="alpha_t")
        self._lr_update_t = tf.convert_to_tensor(self._lr_update, name="lr_update_t")
        self._lr_max_t = tf.convert_to_tensor(self._lr_max, name="lr_max_t")
        self._lr_min_t = tf.convert_to_tensor(self._lr_min, name="lr_min_t")
        self._lr_force_t = tf.convert_to_tensor(self._lr_force, name="lr_force_t")

    def _create_slots(self, var_list):
        for v in var_list:
            self._zeros_slot(v, "old_g", 'g_minus_one')
            lr = self._get_or_make_slot_with_initializer(
                var=v, initializer=self._lr, shape=tf.TensorShape([1]),
                dtype=tf.float32, slot_name='lr', op_name='lr')

            self._lr_variables.append(lr)

    def _apply_dense(self, grad: tf.Tensor, var: tf.Variable) -> tf.Operation:
        """Add ops to apply dense gradients to `var`.

        Args:
            grad: A gradient `Tensor`.
            var: A `Variable` object.

        Returns:
            An `Operation`.
        """

        alpha_t = tf.cast(self._alpha_t, var.dtype.base_dtype)
        lr_update_t = tf.cast(self._lr_update_t, var.dtype.base_dtype)
        lr_max_t = tf.cast(self._lr_max_t, var.dtype.base_dtype)
        lr_min_t = tf.cast(self._lr_min_t, var.dtype.base_dtype)
        lr_force_t = tf.cast(self._lr_force_t, var.dtype.base_dtype)

        old_g = self.get_slot(var, "old_g")
        lr = self.get_slot(var, "lr")

        noise = tf.random_uniform(shape=tf.shape(var), minval=-1.0, maxval=+1.0)

        if self._norm == 'max':
            # compute normalization constant
            g_max = tf.reduce_max(tf.abs(grad))
            denominator = _EPSILON + g_max
            g_update_normed = grad / denominator
        else:
            g_update_normed = tf.nn.l2_normalize(grad)

        # compute update grad
        update_grad = lr * (g_update_normed + noise * alpha_t)
        var_update = tf.assign_sub(var, update_grad)

        # update learning rate
        g_normed = tf.nn.l2_normalize(grad)
        old_g_normed = tf.nn.l2_normalize(old_g)

        lr_change = - lr_update_t * tf.reduce_sum(g_normed * old_g_normed)
        new_lr = lr * (1 - lr_change) + lr_force_t * lr * lr_update_t

        new_lr = tf.clip_by_value(new_lr, lr_min_t, lr_max_t)
        lr_update = tf.assign(lr, new_lr)
        old_g_update = tf.assign(old_g, grad)

        update = tf.group([var_update, lr_update, old_g_update])

        return update

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")

    def _resource_apply_dense(self, grad, handle):
        raise NotImplementedError("Resource apply dense not implemented.")

    def _resource_apply_sparse(self, grad, handle, indices):
        raise NotImplementedError("Resource apply sparce not implemented.")