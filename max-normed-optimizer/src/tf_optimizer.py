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
            momentum: float = 0.0,
            momentum_update: float = 0.0,
            norm: str = 'max',
            noise_amplitude: Union[float, tf.Tensor] = 0.01,
            use_locking: bool = False,
            name: str = "NormalizedSGD"
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

        if norm not in ['max', 'l2', 'std']:
            raise ValueError(f'Unexpected norm type `{norm}`.')

        self._lr = lr
        self._lr_update = lr_update
        self._lr_max = lr_max if lr_max is not None else lr
        self._lr_min = lr_min if lr_min is not None else 0.0
        self._lr_force = lr_force
        self._norm = norm
        self._momentum = momentum
        self._momentum_update = momentum_update
        self._alpha = noise_amplitude

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._alpha_t = None
        self._lr_update_t = None
        self._lr_max_t = None
        self._lr_min_t = None
        self._lr_force_t = None
        self._momentum_t = None
        self._lr_variables = []
        self._m_variables = []

    def _prepare(self):
        self._lr_t = tf.convert_to_tensor(self._lr, name="lr")
        self._alpha_t = tf.convert_to_tensor(self._alpha, name="alpha_t")
        self._lr_update_t = tf.convert_to_tensor(self._lr_update,
                                                 name="lr_update_t")
        self._lr_max_t = tf.convert_to_tensor(self._lr_max, name="lr_max_t")
        self._lr_min_t = tf.convert_to_tensor(self._lr_min, name="lr_min_t")
        self._momentum_t = tf.convert_to_tensor(self._momentum, name="momentum")
        self._lr_force_t = tf.convert_to_tensor(self._lr_force,
                                                name="lr_force_t")

    def _create_slots(self, var_list):
        for v in var_list:
            self._zeros_slot(v, "old_g", 'g_minus_one')
            self._zeros_slot(v, "m", 'momentum')
            lr = self._get_or_make_slot_with_initializer(
                var=v, initializer=self._lr, shape=tf.TensorShape([1]),
                dtype=tf.float32, slot_name='lr', op_name='lr')

            m = self._get_or_make_slot_with_initializer(
                var=v, initializer=self._momentum, shape=tf.TensorShape([1]),
                dtype=tf.float32, slot_name='mm', op_name='mm')
            self._lr_variables.append(lr)
            self._m_variables.append(m)

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
        m = self.get_slot(var, "m")
        lr = self.get_slot(var, "lr")
        m_scalar = self.get_slot(var, "mm")

        noise = tf.random_uniform(shape=tf.shape(var), minval=-1.0, maxval=+1.0)

        averaged_grad = m * m_scalar + grad #* (1 - m_scalar)
        with tf.control_dependencies([averaged_grad]):
            if self._norm == 'max':
                # compute normalization constant
                g_max = tf.reduce_max(tf.abs(averaged_grad))
                denominator = _EPSILON + g_max
                g_update_normed = averaged_grad / denominator
            elif self._norm == 'std':
                std = tf.keras.backend.std(averaged_grad) + _EPSILON
                g_update_normed = averaged_grad / std
            else:
                g_update_normed = tf.nn.l2_normalize(averaged_grad)

        # compute update grad
        update_grad = lr * (g_update_normed + noise * alpha_t)
        var_update = tf.assign_sub(var, update_grad)
        update_m = tf.assign(m, averaged_grad)

        # update learning rate
        g_normed = tf.nn.l2_normalize(grad)
        old_g_normed = tf.nn.l2_normalize(old_g)

        lr_change = - tf.reduce_sum(g_normed * old_g_normed)
        self.test = lr_change
        new_lr = lr * (1 - lr_update_t * lr_change) + lr_force_t * lr
        new_lr = tf.clip_by_value(new_lr, lr_min_t, lr_max_t)

        beta = 1 - self._momentum_update
        new_m = m_scalar * beta + (1 - beta) * lr_change
        new_m = tf.clip_by_value(new_m, 0.0, 1.0)

        with tf.control_dependencies([new_lr, new_m]):
            lr_update = tf.assign(lr, new_lr)
            m_update = tf.assign(m_scalar, new_m)
            old_g_update = tf.assign(old_g, grad)

        update = tf.group([update_m, var_update, lr_update, old_g_update, m_update])

        return update

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")

    def _resource_apply_dense(self, grad, handle):
        raise NotImplementedError("Resource apply dense not implemented.")

    def _resource_apply_sparse(self, grad, handle, indices):
        raise NotImplementedError("Resource apply sparce not implemented.")


class BarzilaiBorweinNormalizedSGD(tf.train.Optimizer):
    """An implementation of normalized Stochastic Gradient Descent.

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
            lr_force: relative force lr to increase in consecutive steps.
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
            # agg_grad = (gk + update_grad)/tf.to_float(steps)

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
                tf.greater_equal(tf.to_float(global_step) / tf.to_float(steps), 2),
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
