import tensorflow as tf
import numpy as np

keras = tf.keras


class SimpleModule(tf.keras.Model):
    def __init__(self):
        super(SimpleModule, self).__init__(name="Module")
        self.num_filters = 3
        self.dense_1 = keras.layers.Dense(self.num_filters, activation="tanh")
        self.dense_2 = keras.layers.Dense(self.num_filters, activation="tanh")

    def call(self, inputs, **kwargs):
        x = self.dense_1(inputs)
        return self.dense_2(x)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_filters
        return tf.TensorShape(shape)


class ODEModel(tf.keras.Model):
    def __init__(self):
        super(ODEModel, self).__init__()
        self.linear1 = keras.layers.Dense(50)
        self.linear2 = keras.layers.Dense(2)

    def call(self, inputs, **kwargs):
        h = inputs ** 3
        h = self.linear1(h, activation="tanh")
        h = self.linear2(h)
        return h

    def compute_output_shape(self, input_shape):
        return input_shape
