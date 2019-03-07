# https://github.com/rtqichen/torchdiffeq/issues/37#issuecomment-469095405
# https://gist.github.com/rtqichen/91924063aa4cc95e7ef30b3a5491cc52

import autograd.numpy as np
from autograd import grad
from autograd.builtins import tuple
from .init import _default_initializer


def get_continuous_normalizing_flow(input_dim, hidden_dim, n_ensemble):
    blocksize = n_ensemble * input_dim
    init_hypernet_params = [
        (_default_initializer((1, hidden_dim)), np.zeros((1, hidden_dim))),
        (_default_initializer((hidden_dim, hidden_dim)), np.zeros((1, hidden_dim))),
        (_default_initializer((hidden_dim, 3 * blocksize + n_ensemble)), np.zeros((1, 3 * blocksize + n_ensemble))),
    ]

    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def sigmoid(x):
        return 1. / (1. + np.exp(-x))

    def _hypernet(t, hypernet_params):
        t = np.array(t).reshape(1, 1)
        params = t
        for i, (w, b) in enumerate(hypernet_params):
            params = np.dot(params, w) + b
            if i < len(hypernet_params) - 1:
                params = np.tanh(params)

        # restructure
        params = params.reshape(-1)
        W = params[:blocksize].reshape(n_ensemble, input_dim, 1)
        U = params[blocksize:2 * blocksize].reshape(n_ensemble, 1, input_dim)
        G = sigmoid(params[2 * blocksize:3 * blocksize]).reshape(n_ensemble, 1, input_dim)
        U = U * G
        B = params[3 * blocksize:].reshape(n_ensemble, 1, 1)
        return [W, B, U]

    def nonlinear(X, W, B):
        return np.tanh(np.matmul(X, W) + B)

    grad_nonlinear = grad(lambda X, W, B: np.sum(nonlinear(X, W, B)))

    def _forward(x, params):
        W, B, U = params

        X = np.repeat(x[None], n_ensemble, 0)

        h = nonlinear(X, W, B)
        dx = np.matmul(h, U).mean(0)

        dHdX = grad_nonlinear(X, W, B)
        dlogpx = -np.matmul(dHdX, np.transpose(U, [0, 2, 1])).mean(0)

        return np.concatenate([dx, dlogpx], 1)

    def odefunc_time_dependent(inputs, params, t=tuple((1,))):
        params = _hypernet(t, params)
        x = inputs[:, :input_dim]
        return _forward(x, params)

    return odefunc_time_dependent, [init_hypernet_params]