# Tensorflow implementation of Neural Ordinary Differential Equations

This is simple implementation of [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366)
paper.

* Example result of probability density transformation using CNFs (two moons dataset).
  The first image shows continuous transformation from unit gaussian to two moons.
  Second image shows training loss, initial samples from two moons and target
  unit gaussian evolution during training.

<img src="img/density.gif" width="256" height="256">
<img src="img/training.gif" width="800" height="256">

## Slides:

Published google slides presentation and other materials:


## Notebooks:

Notebooks are related with the content of the slides

* [0.Implementing_black_box_solver.ipynb](0.Implementing_black_box_solver.ipynb) -
  example which shows how to create simple black-box ODE solver based
  radioactive decay problem.
* [1.Demo_spiral.ipynb](1.Demo_spiral.ipynb) -
  reimplementation of the fitting spiral trajectory problem.
* [1A.Adjoint_method.ipynb](1A.Adjoint_method.ipynb) -
  example which shows how to use adjoint method to optimize initial
  conditions of the ODE to satisfy final conditions.
* [2.Demo_optimize_bullet_trajectory.ipynb](2.Demo_optimize_bullet_trajectory.ipynb) -
  similar to **1A**, problem of finding initial condition for bullet
  velocity **v** in order to hit certain location x. Air drag is included.
* [3.Continuous_Normalizing_Flow.ipynb](3.Continuous_Normalizing_Flow.ipynb) -
  demo of example implementation of Continuous Normalizing Flows (CNFs).
  Fitting of two moons dataset to unit gaussian and planar flows
  reimplementation of the selected paper result.
* [4.Adaptive_forward_errors.ipynb](4.Adaptive_forward_errors.ipynb) -
  demo of example implementation of Continuous Normalizing Flows (CNFs).
  Fitting of two moons dataset to unit gaussian and planar flows
  reimplementation of the selected paper result.


## Requirements

* tensorflow>=1.12

## Solvers


This implementation supports few fixed grid solvers (however adaptive
forward integration mode is also possible with tensorflow odeint call).

Implemented solvers (only explicit):
* Euler
* Midpoint
* RK4

It should be easy to implement any other solver (I believe). For example
this is how RK4 solver is implemented:

```python

def rk4_step(func, dt, state):
    k1 = func(state)
    k2 = func(euler_update(state, k1, dt / 2))
    k3 = func(euler_update(state, k2, dt / 2))
    k4 = func(euler_update(state, k3, dt))

    return zip_map(
        zip(state, k1, k2, k3, k4),
        lambda h, dk1, dk2, dk3, dk4:
            h + dt * (dk1 + 2 * dk2 + 2 * dk3 + dk4) / 6,
    )
```

Reverse mode is implemented with adjoint method or can be computed
with explicit backpropagation through graph.


## Examples

### Forward and reverse mode with Keras API

Example usage with Keras API:

```python
import neural_ode

class NNModule(tf.keras.Model):
    def __init__(self, num_filters):
        super(NNModule, self).__init__(name="Module")
        self.dense_1 = Dense(num_filters, activation="tanh")
        self.dense_2 = Dense(num_filters, activation="tanh")

    def call(self, inputs, **kwargs):
        t, x = inputs
        h = self.dense_1(x)
        return self.dense_2(h)

# create model and ode solver
model = NNModule(num_filters=3)
ode = NeuralODE(
    model, t=np.linspace(0, 1.0, 20),
    solver=neural_ode.rk4_step
)
x0 = tf.random_normal(shape=[12, 3])

with tf.GradientTape() as g:
    g.watch(x0)
    xN = ode.forward(x0)
    # some loss function here e.g. L2
    loss = xN ** 2

# reverse mode with adjoint method
dLoss = g.gradient(loss, xN)
x0_rec, dLdx0, dLdW = ode.backward(xN, dLoss)
# this is equivalent to:
dLdx0, *dLdW = g.gradient(loss, [x0, *model.weights])
```

## CNFs
For full and optimized code see notebook about CNFs:

```python

num_samples = 512
cnf_net = CNF(input_dim=2, hidden_dim=32, n_ensemble=16)
ode = NeuralODE(model=cnf_net, t=np.linspace(0, 1, 10))

for _ in ranage(1000):
    x0 = tf.to_float(make_moons(n_samples=num_samples, noise=0.08)[0])
    logdet0 = tf.zeros([num_samples, 1])
    h0 = tf.concat([x0, logdet0], axis=1)

    hN = ode.forward(inputs=h0)
    with tf.GradientTape() as g:
        g.watch(hN)
        xN, logdetN = hN[:, :2], hN[:, 2]
        mle = tf.reduce_sum(p0.log_prob(xN), -1)
        loss = - tf.reduce_mean(mle - logdetN)

    h0_rec, dLdh0, dLdW = ode.backward(hN, g.gradient(loss, hN))
    optimizer.apply_gradients(zip(dLdW, cnf_net.weights))
```

## Optimizations
One can convert NeuralODE calls to static graph with **tf.function**
(tf>=1.12) or **defunc** (tf 1.12). Here is example:

```python
from neural_ode import defun_neural_ode
ode = NeuralODE(model=some_model, t=t_grid, solver=some_solver)
ode = defun_neural_ode(ode)
...
# first call will take some time
outputs = ode.forward(inputs=some_inputs)
# second call will be much faster
outputs = ode.forward(inputs=some_inputs)
```





## Other:
For more examples see `test_neural_ode.py` file.


## Links

* CNF autograd https://gist.github.com/rtqichen/91924063aa4cc95e7ef30b3a5491cc52
* CNF reference implementation (see link above): [cnf_autograd_python.py](cnf_autograd_python.py)
* Original implementation in pytorch: [torchdiffeq](https://github.com/rtqichen/torchdiffeq)

## Limitations

* no adaptive solver for reverse mode
* no implicit solvers
* tensorflow eager mode seems to be slow even after
  functionalization of the graph

