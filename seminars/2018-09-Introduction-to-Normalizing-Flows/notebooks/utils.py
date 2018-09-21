import numpy as np
import tensorflow as tf
from typing import Optional, Callable
from tensorflow.python.layers import core as layers
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict


def numpy_array_to_dataset(
        array: np.array,
        buffer_size: int = 512,
        batch_size: int = 100,
        num_parallel_batches: int = 16,
        preprocess_fn: Optional[Callable] = None,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(array.astype(np.float32))
    dataset = dataset.apply(
        tf.contrib.data.shuffle_and_repeat(buffer_size=buffer_size, count=-1)
    )
    if preprocess_fn is None:
        dataset = dataset.batch(batch_size=batch_size)
    else:
        dataset = dataset.apply(
            tf.contrib.data.map_and_batch(
                map_func=preprocess_fn,
                batch_size=batch_size,
                num_parallel_batches=num_parallel_batches,
                drop_remainder=True
            )
        )
    dataset = dataset.prefetch(4)
    return dataset


_epsilon = 1e-5


def safe_log(x: tf.Tensor) -> tf.Tensor:
    return tf.log(tf.maximum(x, _epsilon))


def create_autoregressive_masks(
        input_size: int,
        hidden_sizes: list = [500],
        natural_ordering: bool = True,
        seed: int = 4198721
) -> list:
    """Creates list of MADE masks. This implementation was taken from:
    https://github.com/karpathy/pytorch-made
    """

    rng = np.random.RandomState(seed)
    L = len(hidden_sizes)
    m = {}
    # sample the order of the inputs and the connectivity of all neurons
    m[-1] = np.arange(input_size) if natural_ordering else rng.permutation(
        input_size)
    for l in range(L):
        m[l] = rng.randint(m[l - 1].min(), input_size - 1, size=hidden_sizes[l])

    # construct the mask matrices
    masks = [(m[l - 1][:, None] <= m[l][None, :]) * 1 for l in range(L)]
    masks.append((m[L - 1][:, None] < m[-1][None, :]) * 1)
    return masks


def masked_dense(inputs: tf.Tensor,
                 units: int,
                 mask: np.ndarray,
                 activation=None,
                 kernel_initializer=None,
                 reuse=None,
                 name=None,
                 *args,
                 **kwargs) -> tf.Tensor:
    """This code has been copied from masked_dense implementation in
    Tensorflow. See TF documentation:
    https://www.tensorflow.org/api_docs/python/tf/contrib/distributions/bijectors/masked_dense

    """

    input_depth = inputs.shape.with_rank_at_least(1)[-1].value
    if input_depth is None:
        raise NotImplementedError(
            "Rightmost dimension must be known prior to graph execution.")

    if kernel_initializer is None:
        kernel_initializer = init_ops.glorot_normal_initializer()

    def masked_initializer(shape, dtype=None, partition_info=None):
        return mask * kernel_initializer(shape, dtype, partition_info)

    with ops.name_scope(name, "masked_dense", [inputs, units]):
        layer = layers.Dense(
            units,
            activation=activation,
            kernel_initializer=masked_initializer,
            kernel_constraint=lambda x: mask * x,
            name=name,
            dtype=inputs.dtype.base_dtype,
            _scope=name,
            _reuse=reuse,
            *args,
            **kwargs)
        return layer.apply(inputs)


class Metrics:
    def __init__(self, step, metrics_tensors):
        self.metrics = defaultdict(list)
        self.step = step
        self.metrics_tensors = metrics_tensors

    def check_step(self, i):
        return ((i + 1) % self.step == 0)

    def append(self, results):
        for k, t in self.metrics_tensors.items():
            self.metrics[k].append(results[k])

    def get(self):
        return self.metrics_tensors

    @property
    def num_metrics(self):
        return len(self.metrics)


class PlotMetricsHook():
    def __init__(self, metrics: Metrics, step=1000, figsize=(15, 3),
                 skip_steps=5):
        self.metrics = metrics
        self.step = step
        self.figsize = figsize
        self.skip_steps = skip_steps

    def check_step(self, i):
        return ((i + 1) % self.step == 0)

    def run(self):
        plt.figure(figsize=self.figsize)

        for k, (m, values) in enumerate(self.metrics.metrics.items()):
            plt.subplot(1, self.metrics.num_metrics, k + 1)
            plt.title(m)
            vals = values[self.skip_steps:]
            plt.plot(vals)
            vals = np.array(vals)
            if len(vals) > 0:
                plt.ylim([vals.min(), vals.max()])
        plt.show()


def trainer(sess, num_steps, train_op, feed_dict_fn, metrics, hooks):
    for i in tqdm(range(num_steps)):
        fetches = {"train_op": train_op}

        for metric in metrics:
            if metric.check_step(i):
                fetches.update(**metric.get())

        results = sess.run(fetches=fetches, feed_dict=feed_dict_fn())

        for metric in metrics:
            if metric.check_step(i):
                metric.append(results)

        for hook in hooks:
            if hook.check_step(i):
                hook.run()


def plot_4x4_grid(images: np.ndarray, shape: tuple=(28, 28), cmap='gray'):
    """
    Plot multiple images in subplot grid.
    :param images: tensor with MNIST images with shape [16, *shape]
    :param shape: shape of the images
    """
    assert images.shape[0] >= 16
    dist_samples_np = images[:16, ...].reshape([4, 4, *shape])

    plt.figure(figsize=(4, 4))
    for i in range(4):
        for j in range(4):
            plt.subplot(4, 4, i * 4 + j + 1)
            plt.imshow(dist_samples_np[i, j], cmap=cmap)
            plt.xticks([])
            plt.yticks([])
    plt.subplots_adjust(hspace=0.05, wspace=0.05)