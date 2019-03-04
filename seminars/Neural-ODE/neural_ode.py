from typing import Optional
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
import tensorflow.contrib.eager as tfe
keras = tf.keras


def zip_map(zipped, update_op):
    return [update_op(*elems) for elems in zipped]


def euler_update(h_list, dh_list, dt):
    return zip_map(zip(h_list, dh_list), lambda h, dh: h + dt * dh)


def euler_step(func, dt, state):
    return euler_update(state, func(state), dt)


def rk2_step(func, dt, state):
    k1 = func(state)
    k2 = func(euler_update(state, k1, dt))
    return zip_map(
        zip(state, k1, k2),
        lambda h, dk1, dk2: h + dt * (dk1 + dk2) / 2
    )


def rk4_step(func, dt, state):
    k1 = func(state)
    k2 = func(euler_update(state, k1, dt / 2))
    k3 = func(euler_update(state, k2, dt / 2))
    k4 = func(euler_update(state, k3, dt))

    return zip_map(
        zip(state, k1, k2, k3, k4),
        lambda h, dk1, dk2, dk3, dk4: h + dt * (dk1 + 2 * dk2 + 2 * dk3 + dk4) / 6
    )


class NeuralODE:
    def __init__(self, model: tf.keras.Model, t=np.linspace(0, 1, 40),
                 solver=euler_step):
        self._t = t
        self._model = model
        self._solver = solver
        self._deltas_t = t[1:] - t[:-1]

    def forward(self, inputs: tf.Tensor, return_states: Optional[str] = None):
        assert type(inputs) in (tf.Tensor, EagerTensor), \
            f"inputs must be Tensor but is: {type(inputs)}"

        def _forward_dynamics(state):
            return [1.0, self._model(inputs=state)]

        states = []
        t0 = tf.to_float(self._t[0])
        state = [t0, inputs]

        for dt in self._deltas_t:
            state = self._solver(
                func=_forward_dynamics,
                dt=tf.to_float(dt),
                state=state
            )
            if return_states == "numpy":
                states.append(state[1].numpy())
            elif return_states == "tf":
                states.append(state[1])

        if return_states:
            return state[1], states
        return state[1]

    def _backward_dynamics(self, state):
        t = state[0]
        ht = state[1]
        at = - state[2]

        with tf.GradientTape() as g:
            g.watch(ht)
            ht_new = self._model(inputs=[t, ht])

        gradients = g.gradient(
            target=ht_new,
            sources=[ht] + self._model.weights,
            output_gradients=at
        )
        return [1.0, ht_new, *gradients]

    def backward(self, outputs: tf.Tensor, output_gradients: tf.Tensor):
        assert type(outputs) in (tf.Tensor, EagerTensor), \
            f"outputs must be Tensor but is: {type(outputs)}"

        grad_weights = [tf.zeros_like(w) for w in self._model.weights]
        t0 = tf.to_float(self._t[-1])
        state = [t0, outputs, output_gradients, *grad_weights]
        for dt in self._deltas_t[::-1]:
            state = self._solver(
                self._backward_dynamics,
                dt=- tf.to_float(dt),
                state=state
            )

        return state[1], state[2], state[3:]


def defun_neural_ode(node: NeuralODE) -> NeuralODE:
    node.forward = tfe.defun(node.forward)
    node.backward = tfe.defun(node.backward)
    return node