"""Utility functions used in notebooks"""
from collections import defaultdict
from typing import Optional, Callable

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def numpy_array_to_dataset(
        array: np.array,
        buffer_size: int = 512,
        batch_size: int = 100,
        num_parallel_batches: int = 16,
        preprocess_fn: Optional[Callable] = None,
) -> tf.data.Dataset:
    """Convert numpy array to tf.data.Dataset"""
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
                drop_remainder=True,
            )
        )
    dataset = dataset.prefetch(4)
    return dataset


class Metrics:
    def __init__(self, step, metrics_tensors):
        self.metrics = defaultdict(list)
        self.step = step
        self.metrics_tensors = metrics_tensors

    def check_step(self, i):
        return (i + 1) % self.step == 0

    def append(self, results):
        for k, t in self.metrics_tensors.items():
            self.metrics[k].append(results[k])
            print(k, results[k])

    def get(self):
        return self.metrics_tensors

    @property
    def num_metrics(self):
        return len(self.metrics)


class PlotMetricsHook:
    def __init__(self, metrics: Metrics, step=1000, figsize=(15, 3),
                 skip_steps=5):
        self.metrics = metrics
        self.step = step
        self.figsize = figsize
        self.skip_steps = skip_steps

    def check_step(self, i):
        return (i + 1) % self.step == 0

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
