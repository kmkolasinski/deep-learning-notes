from typing import Union, Optional, Callable

import tensorflow as tf

_EPSILON = 1e-6


def max_normalize(grad: tf.Tensor) -> tf.Tensor:
    g_max = tf.reduce_max(tf.abs(grad))
    norm = _EPSILON + g_max
    return grad / norm


def std_normalize(grad: tf.Tensor) -> tf.Tensor:
    std = tf.keras.backend.std(grad) + _EPSILON
    return grad / std


def l2_normalize(grad: tf.Tensor) -> tf.Tensor:
    return tf.nn.l2_normalize(grad)


def avg_l2_normalize(grad: tf.Tensor) -> tf.Tensor:
    norm = tf.sqrt(tf.reduce_mean(tf.square(grad))) + _EPSILON
    return grad / norm


_NORMS = {
    'max': max_normalize,
    'l2': l2_normalize,
    'std': std_normalize,
    'avg_l2': avg_l2_normalize,
}


class NormalizedSGD(tf.train.Optimizer):
    """A simplified implementation of normalized Stochastic gradient descent.
    In every step gradient `Grad` is normalized by selected norm.

    @@__init__
    """
    def __init__(
            self,
            lr: Union[float, tf.Tensor],
            norm: Union[str, Callable[[tf.Tensor], tf.Tensor]],
            use_locking: bool = False,
            name: str = "NormalizedSGD"
    ) -> None:
        """Optimizer constructor.

        Args:
            lr: learning rate value or tensor
            norm: one of 'max', 'l2', 'std', 'avg_l2' or callable which takes
                gradient and returns a normalized gradient
            use_locking: Bool. If True apply use locks to prevent concurrent
                updates to variables.
            name: A non-empty string.  The name to use for accumulators created
                for the optimizer.
        """
        super(NormalizedSGD, self).__init__(use_locking, name)
        self._lr = lr
        if type(norm) == str:
            if norm not in _NORMS:
                raise ValueError(f'Provided norm `{norm}` must'
                                 f' be one of `{_NORMS.keys()}`.')
            self._norm_fn = _NORMS[norm]
        elif callable(norm):
            self._norm_fn = norm
        else:
            raise ValueError('Norm must be either a string or callable')

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None

    def _prepare(self):
        self._lr_t = tf.convert_to_tensor(self._lr, name="learning_rate")

    def _apply_dense(self, grad: tf.Tensor, var: tf.Variable) -> tf.Operation:
        """Add ops to apply dense gradients to `var`.

        Args:
            grad: A gradient `Tensor`.
            var: A `Variable` object.

        Returns:
            An `Operation`.
        """
        lr_t = tf.cast(self._lr_t, var.dtype.base_dtype)
        update_grad = lr_t * self._norm_fn(grad)

        return tf.assign_sub(var, update_grad)


class LmaxNormalizedSGD(tf.train.Optimizer):
    """A simplified implementation of normalized Stochastic gradient descent.
    In every step gradient `Grad` is normalized by:
         L_inf = max(abs(Grad))

    @@__init__
    """

    def __init__(
            self,
            lr: Union[float, tf.Tensor],
            noise_amplitude: Union[float, tf.Tensor] = 0.0,
            use_locking: bool = False,
            name: str = "LmaxNormalizedSGD"
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
            use_locking: Bool. If True apply use locks to prevent concurrent
                updates to variables.
            name: A non-empty string.  The name to use for accumulators created
                for the optimizer.
        """
        super(LmaxNormalizedSGD, self).__init__(use_locking, name)
        self._lr = lr
        self._alpha = noise_amplitude

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._alpha_t = None

    def _prepare(self):
        self._lr_t = tf.convert_to_tensor(self._lr, name="learning_rate")
        self._alpha_t = tf.convert_to_tensor(self._alpha, name="alpha_t")

    def _apply_dense(self, grad: tf.Tensor, var: tf.Variable) -> tf.Operation:
        """Add ops to apply dense gradients to `var`.

        Args:
            grad: A gradient `Tensor`.
            var: A `Variable` object.

        Returns:
            An `Operation`.
        """
        lr_t = tf.cast(self._lr_t, var.dtype.base_dtype)
        alpha_t = tf.cast(self._alpha_t, var.dtype.base_dtype)
        noise = tf.random_uniform(shape=tf.shape(var), minval=-1.0, maxval=+1.0)

        # compute normalization constant
        g_max = tf.reduce_max(tf.abs(grad))
        normalization = _EPSILON + g_max
        update_grad = lr_t * (grad / normalization + noise * alpha_t)

        return tf.assign_sub(var, update_grad)

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")


class AdaptiveNormalizedSGD(tf.train.Optimizer):
    """An implementation of normalized Stochastic Gradient Descent.
    With adaptive momentum term.
    @@__init__
    """

    def __init__(
            self,
            lr: Union[float, tf.Tensor],
            lr_update: float = 0.005,
            lr_max: Optional[float] = 0.01,
            lr_min: Optional[float] = 1e-6,
            momentum: float = 0.0,
            momentum_update: float = 0.0,
            norm_type: str = 'max',
            noise_amplitude: Union[float, tf.Tensor] = 0.0,
            use_locking: bool = False,
            name: str = "AdaptiveNormalizedSGD"
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
            momentum: an initial value of momentum,
            momentum_update: adaptive momentum change step size. if 0 momentum
                will remain constant.
            norm_type: a string, one of:
                'max' - normalize gradients of each layer by Lmax norm
                'L2' - normalize gradients of each layer by L2 norm
                'std' - normalize gradients to have unit variance
            noise_amplitude: the amplitude of the noise to be added
                to the normalized gradients. This should help to escape
                flat regions. Noise is added with amplitude which is
                proportional to the learning rate i.e.
                         final_amplitude = noise_amplitude * learning_rate
                Noise is sampled from uniform distribution U[-1, 1].
            use_locking: Bool. If True apply use locks to prevent concurrent
                updates to variables.
            name: A non-empty string. The name to use for accumulators created
                for the optimizer.
        """
        super(AdaptiveNormalizedSGD, self).__init__(use_locking, name)

        if norm_type not in ['max', 'l2', 'std']:
            raise ValueError(f'Unexpected norm type `{norm_type}`.')

        self._lr = lr
        self._lr_update = lr_update
        self._lr_max = lr_max if lr_max is not None else lr
        self._lr_min = lr_min if lr_min is not None else 0.0
        self._norm_type = norm_type
        self._momentum_coef = momentum
        self._momentum_coef_update = momentum_update
        self._alpha = noise_amplitude

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._alpha_t = None
        self._lr_update_t = None
        self._lr_max_t = None
        self._lr_min_t = None
        self._momentum_coef_t = None
        self._momentum_coef_update_t = None
        self._grad_correlation_t = None
        self._lr_variables = []
        self._momentum_variables = []

    def _prepare(self):

        self._alpha_t = tf.convert_to_tensor(
            self._alpha, name="alpha")
        self._lr_update_t = tf.convert_to_tensor(
            self._lr_update, name="lr_update")
        self._lr_max_t = tf.convert_to_tensor(
            self._lr_max, name="lr_max")
        self._lr_min_t = tf.convert_to_tensor(
            self._lr_min, name="lr_min")
        self._momentum_coef_update_t = tf.convert_to_tensor(
            self._momentum_coef_update, name="momentum_coef_update")

    def _create_slots(self, var_list):
        for v in var_list:
            self._zeros_slot(v, "old_grad", 'old_grad')
            self._zeros_slot(v, "momentum", 'momentum')

            lr = self._get_or_make_slot_with_initializer(
                var=v, initializer=self._lr, shape=tf.TensorShape([1]),
                dtype=tf.float32, slot_name='lr', op_name='lr')

            m_coef = self._get_or_make_slot_with_initializer(
                var=v, initializer=self._momentum_coef,
                shape=tf.TensorShape([1]),
                dtype=tf.float32, slot_name='m_coef', op_name='m_coef')
            self._lr_variables.append(lr)
            self._momentum_variables.append(m_coef)

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
        m_coef_update_t = tf.cast(self._momentum_coef_update_t,
                                  var.dtype.base_dtype)

        # get cached tensors
        old_grad = self.get_slot(var, "old_grad")
        momentum = self.get_slot(var, "momentum")
        # learnable stuff
        lr = self.get_slot(var, "lr")
        m_coef = self.get_slot(var, "m_coef")

        # generate random noise
        noise = alpha_t * tf.random_uniform(
            shape=tf.shape(var), minval=-1.0, maxval=+1.0)

        # compute aggregated gradient
        momentum_grad = momentum * m_coef + grad
        with tf.control_dependencies([momentum_grad]):
            if self._norm_type == 'max':
                # compute normalization constant
                g_max = tf.reduce_max(tf.abs(momentum_grad))
                denominator = _EPSILON + g_max
                g_update_normed = momentum_grad / denominator
            elif self._norm_type == 'std':
                std = tf.keras.backend.std(momentum_grad) + _EPSILON
                g_update_normed = momentum_grad / std
            else:
                g_update_normed = tf.nn.l2_normalize(momentum_grad)

        # compute update grad
        update_grad = lr * (g_update_normed + noise)
        var_update = tf.assign_sub(var, update_grad)
        update_m = tf.assign(momentum, momentum_grad)

        # compute gradient correlation
        g_normed = tf.nn.l2_normalize(grad)
        old_g_normed = tf.nn.l2_normalize(old_grad)
        lr_change = - tf.reduce_sum(g_normed * old_g_normed)

        # update learning rate
        new_lr = lr * (1 - lr_update_t * lr_change)
        new_lr = tf.clip_by_value(new_lr, lr_min_t, lr_max_t)

        # update momentum
        beta = 1 - m_coef_update_t
        new_m_coef = m_coef * beta + (1 - beta) * lr_change
        new_m_coef = tf.clip_by_value(new_m_coef, 0.0, 1.0)

        self._grad_correlation_t = lr_change

        with tf.control_dependencies([new_lr, new_m_coef]):
            lr_update = tf.assign(lr, new_lr)
            m_update = tf.assign(m_coef, new_m_coef)
            old_g_update = tf.assign(old_grad, grad)

        return tf.group([update_m, var_update,
                         lr_update, old_g_update, m_update])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")

    def _resource_apply_dense(self, grad, handle):
        raise NotImplementedError("Resource apply dense not implemented.")

    def _resource_apply_sparse(self, grad, handle, indices):
        raise NotImplementedError("Resource apply sparce not implemented.")


class BarzilaiBorweinNormalizedSGD(tf.train.Optimizer):
    """An implementation of normalized Stochastic Gradient Descent.
    Based on idea in:
        "Barzilai-Borwein Step Size for Stochastic Gradient Descent",
        Conghui Tan, Shiqian Ma, Yu-Hong Dai, Yuqiu Qian (2016)
        https://arxiv.org/abs/1605.04131

    Note: This implementation is probably incorrect.

    @@__init__
    """

    def __init__(
            self,
            lr: Union[float, tf.Tensor],
            lr_update: float = 0.01,
            lr_max: Optional[float] = 0.01,
            lr_min: Optional[float] = 1e-5,
            steps: int = 10,
            norm: str = 'max',
            noise_amplitude: Union[float, tf.Tensor] = 0.01,
            use_locking: bool = False,
            name: str = "BarzilaiBorweinNormalizedSGD"
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
            use_locking: Bool. If True apply use locks to prevent concurrent
                updates to variables.
            name: A non-empty string. The name to use for accumulators created
                for the optimizer.
        """
        super(BarzilaiBorweinNormalizedSGD, self).__init__(use_locking, name)

        if norm not in ['max', 'l2']:
            raise ValueError(f'Unexpected norm type `{norm}`.')

        self._lr = lr
        self._lr_update = lr_update
        self._lr_max = lr_max if lr_max is not None else lr
        self._lr_min = lr_min if lr_min is not None else 0.0
        self._steps = steps
        self._norm = norm
        self._alpha = noise_amplitude

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._alpha_t = None
        self._lr_update_t = None
        self._lr_max_t = None
        self._lr_min_t = None
        self._steps_t = None
        self._lr_variables = []
        self._current_step = None

    def _prepare(self):
        self._lr_t = tf.convert_to_tensor(self._lr, name="lr")
        self._alpha_t = tf.convert_to_tensor(self._alpha, name="alpha_t")
        self._lr_update_t = tf.convert_to_tensor(self._lr_update,
                                                 name="lr_update_t")
        self._lr_max_t = tf.convert_to_tensor(self._lr_max, name="lr_max_t")
        self._lr_min_t = tf.convert_to_tensor(self._lr_min, name="lr_min_t")
        self._steps_t = tf.convert_to_tensor(
            self._steps, name="steps", preferred_dtype=tf.int32)

    def _create_slots(self, var_list):

        first_var = min(var_list, key=lambda x: x.name)
        self._current_step = self._create_non_slot_variable(
            initial_value=0,
            name="current_step",
            colocate_with=first_var)
        self._global_step = self._create_non_slot_variable(
            initial_value=0,
            name="global_step",
            colocate_with=first_var)

        for v in var_list:
            self._zeros_slot(v, "gk_old", 'gk_old')
            self._zeros_slot(v, "v_old", 'v_old')
            self._zeros_slot(v, "gk", 'gk')
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
        steps = self._steps_t

        current_step = self._current_step
        global_step = self._global_step

        gk_old = self.get_slot(var, "gk_old")
        gk = self.get_slot(var, "gk")
        var_old = self.get_slot(var, "v_old")
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

        beta = 0.9

        def update_grads():

            agg_grad = gk * beta + (1 - beta) * update_grad
            # agg_grad = gk + update_grad
            update_gk = tf.assign(gk, agg_grad)

            return tf.group([update_gk]), lr

        def reset_steps():

            agg_grad = gk * beta + (1 - beta) * update_grad
            # I did try it however it was not stable :/
            # dx = var - var_old
            # dg = gk - gk_old
            # s1 = tf.reduce_sum(tf.square(dx))
            # s2 = tf.abs(tf.reduce_sum(dx * dg)) + _EPSILON
            # eta = s1 / s2

            # update learning rate
            g_normed = tf.nn.l2_normalize(agg_grad)
            old_g_normed = tf.nn.l2_normalize(gk_old)
            lr_change = - lr_update_t * tf.reduce_sum(g_normed * old_g_normed)
            eta = lr * (1 - lr_change)

            with tf.control_dependencies([eta]):
                update_gk_old = tf.assign(gk_old, agg_grad)
                with tf.control_dependencies([update_gk_old]):
                    update_gk = tf.assign(gk, tf.zeros_like(gk))

                update_var_old = tf.assign(var_old, var)
                step_assign = tf.assign(current_step, 0)
                update_g = tf.group([update_gk_old,
                                     update_var_old,
                                     update_gk,
                                     step_assign])

            return update_g, eta

        with tf.control_dependencies([var_update]):
            udaptes, new_lr = tf.cond(tf.greater_equal(current_step, steps),
                                      true_fn=reset_steps,
                                      false_fn=update_grads)

        with tf.control_dependencies([udaptes]):
            new_lr = tf.cond(
                tf.greater_equal(tf.to_float(global_step) / tf.to_float(steps),
                                 2),
                true_fn=lambda: new_lr,
                false_fn=lambda: lr
            )

        global_step_update = tf.assign_add(global_step, 1)
        step_update = tf.assign_add(current_step, 1)

        new_lr = tf.clip_by_value(new_lr, lr_min_t, lr_max_t)
        lr_update = tf.assign(lr, new_lr)

        update = tf.group([lr_update, step_update, global_step_update])

        return update

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")

    def _resource_apply_dense(self, grad, handle):
        raise NotImplementedError("Resource apply dense not implemented.")

    def _resource_apply_sparse(self, grad, handle, indices):
        raise NotImplementedError("Resource apply sparce not implemented.")
