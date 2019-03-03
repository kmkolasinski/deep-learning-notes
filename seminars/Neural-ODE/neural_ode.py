from typing import Optional

import tensorflow as tf
from tensorflow.python.keras import Model
import tensorflow.contrib.eager as tfe
keras = tf.keras


def euler_update(h_list, dh_list, dt):
    outputs = []
    for h, dh in zip(h_list, dh_list):
        if type(h) == list:
            outs = [hp + dt * dhp for hp, dhp in zip(h, dh)]
        else:
            outs = h + dt * dh
        outputs.append(outs)

    return outputs #[h + dt * dh for h, dh in zip(h_list, dh_list)]


def euler_step(fun, state, dt):
    dstate = fun(state)
    return euler_update(state, dstate, dt)


def rk2_step(fun, state, dt):
    k1 = fun(state)
    k2 = fun(euler_update(state, k1, dt))

    outputs = []
    for hprim, k1prim, k2prim in zip(state, k1, k2):
        if type(hprim) == list:
            outs = [hbis + dt * (k1bis + k2bis) / 2 for hbis, k1bis, k2bis in zip(hprim, k1prim, k2prim)]
        else:
            outs = hprim + dt * (k1prim + k2prim) / 2

        outputs.append(outs)
    return outputs #[h + dt * (a + b) / 2 for h, a, b in zip(state, k1, k2)]


class NeuralODE:
    def __init__(self, module_fn: Model, solver=rk2_step, num_steps: int = 40, tmax: float = 1.0):
        self._num_steps = num_steps
        self._dt = tmax / num_steps
        self._model = module_fn
        self._solver = solver

    def forward(self, h_input, return_states: Optional[str] = None):

        def _forward_dynamics(h):
            return [self._model(h[0])]

        state = [h_input]
        if return_states == "numpy":
            states = [h_input.numpy()]
        elif return_states == "tf":
            states = [h_input]

        for k in range(self._num_steps):
            state = self._solver(_forward_dynamics, state, self._dt)
            if return_states == "numpy":
                states.append(state[0].numpy())
            elif return_states == "tf":
                states.append(state[0])

        if return_states:
            return state[0], states
        return state[0]

    def _backward_dynamics(self, state):
        ht = state[0]
        at = - state[1]

        with tf.GradientTape() as g:
            g.watch(ht)
            ht_new = self._model(ht)

        gradients = g.gradient(
            target=ht_new,
            sources=[ht] + self._model.weights,
            output_gradients=at
        )
        return [ht_new, *gradients]

    def backward(self, h_output, grad_h_output):

        dWeights = [tf.zeros_like(w) for w in self._model.weights]
        neg_dt = - self._dt

        state = [h_output, grad_h_output, *dWeights]
        for k in range(self._num_steps):
            state = self._solver(self._backward_dynamics, state, neg_dt)

        return state[0], state[1], state[2:]

    def defun(self):
        self._model.call = tfe.defun(self._model.call)
        self.forward = tfe.defun(self.forward)
        self.backward = tfe.defun(self.backward)
