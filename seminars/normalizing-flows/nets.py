from typing import List

import flow_layers as fl
import tensorflow as tf
import tensorflow as tf
import numpy as np
from tensorflow.contrib import framework as tf_framework

K = tf.keras.backend
keras = tf.keras
import tensorflow.contrib.layers as tf_layers
from tensorflow.python.ops import template as template_ops


def simple_resnet_template_fn(name: str):
    def _shift_and_log_scale_fn(x: tf.Tensor):
        shape = K.int_shape(x)
        num_channels = shape[3]
        # nn definition
        h = tf_layers.conv2d(x, num_outputs=num_channels, kernel_size=3,
                             activation_fn=tf.nn.leaky_relu)
        h = tf_layers.conv2d(h, num_outputs=num_channels // 2, kernel_size=3,
                             activation_fn=tf.nn.leaky_relu)
        h = tf_layers.conv2d(h, num_outputs=num_channels, kernel_size=3,
                             activation_fn=None)
        h = tf.nn.leaky_relu(h + x)

        # create shift and log_scale
        shift = tf_layers.conv2d(
            h, num_outputs=num_channels,
            weights_initializer=tf.random_normal_initializer(stddev=0.001),
            kernel_size=3, activation_fn=None,
        )
        log_scale = tf_layers.conv2d(
            h, num_outputs=num_channels,
            weights_initializer=tf.random_normal_initializer(stddev=0.001),
            kernel_size=3, activation_fn=None
        )
        log_scale = tf.clip_by_value(log_scale, -15.0, 15.0)
        return shift, log_scale

    return template_ops.make_template(name, _shift_and_log_scale_fn)


def step_flow(name: str, shift_and_log_scale_fn):
    layers = [
        fl.ActnormLayer(),
        fl.InvertibleConv1x1Layer(),
        fl.AffineCouplingLayer(
            shift_and_log_scale_fn=shift_and_log_scale_fn
        )
    ]
    return [fl.ChainLayer(layers, name=name)]


def create_simple_flow(num_steps: int = 1, num_scales: int=3) -> List[fl.FlowLayer]:
    layers = [
        fl.LogitifyImage(),
    ]

    for scale in range(num_scales):
        scale_name = f"Scale{scale+1}"

        scale_steps = []
        for s in range(num_steps):
            name = f"Step{s+1}"
            scale_steps += step_flow(
                name, simple_resnet_template_fn(name)
            )

        layers += [
            fl.SqueezingLayer(name=scale_name),
            fl.ChainLayer(scale_steps, name=scale_name),
            fl.FactorOutLayer(name=scale_name)
        ]

    return layers
