from typing import Optional

import tensorflow as tf
from tensorflow.python.keras import Model
import tensorflow.contrib.eager as tfe

keras = tf.keras
import numpy as np


def euler_update(h_list, dh_list, dt):
    outputs = []
    for h, dh in zip(h_list, dh_list):
        if type(h) == list:
            outs = [hp + dt * dhp for hp, dhp in zip(h, dh)]
        else:
            outs = h + dt * dh
        outputs.append(outs)

    return outputs


def euler_step(func, t, dt, state):
    dstate = func(t, state)
    return euler_update(state, dstate, dt)


def rk2_step(func, t, dt, state):
    k1 = func(t, state)
    k2 = func(t, euler_update(state, k1, dt))

    outputs = []
    for hprim, k1prim, k2prim in zip(state, k1, k2):
        if type(hprim) == list:
            outs = [hbis + dt * (k1bis + k2bis) / 2 for hbis, k1bis, k2bis in
                    zip(hprim, k1prim, k2prim)]
        else:
            outs = hprim + dt * (k1prim + k2prim) / 2

        outputs.append(outs)
    return outputs


class NeuralODE:
    def __init__(self, model: tf.keras.Model, t=np.linspace(0, 1, 40),
                 solver=euler_step):
        self._t = t
        self._model = model
        self._solver = solver
        self._deltas_t = t[1:] - t[:-1]

    def forward(self, h_input, return_states: Optional[str] = None):

        def _forward_dynamics(t, state):
            return [self._model(inputs=[t, state[0]])]

        if return_states == "numpy":
            states = [h_input.numpy()]
        elif return_states == "tf":
            states = [h_input]

        state = [h_input]
        for dt, t in zip(self._deltas_t, self._t):

            state = self._solver(
                func=_forward_dynamics,
                t=tf.to_float(t),
                dt=tf.to_float(dt),
                state=state
            )

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

        state = [h_output, grad_h_output, *dWeights]
        for dt, t in zip(self._deltas_t[::-1], self._t[::-1]):
            state = self._solver(self._backward_dynamics, state, -dt)

        return state[0], state[1], state[2:]

    # def defun(self):
    #     self._model.call = tfe.defun(self._model.call)
    #     self.forward = tfe.defun(self.forward)
    #     self.backward = tfe.defun(self.backward)
