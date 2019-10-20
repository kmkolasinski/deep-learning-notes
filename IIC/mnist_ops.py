"""
This code has been copied from: https://github.com/astirn/IIC

MIT License

Copyright (c) 2019 astirn

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. © 2019 GitHub, Inc.

"""
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def mnist_x_aug_fn(x_orig, mdl_input_dims = (24, 24, 1), is_training = True):
    x_orig = tf.expand_dims(x_orig, 0)
    # get common shapes
    height_width = mdl_input_dims[:-1]
    n_chans = mdl_input_dims[-1]

    # training transformations
    if is_training:
        x1 = tf.image.central_crop(x_orig, np.mean(20 / np.array(x_orig.shape.as_list()[1:-1])))        
        x2 = tf.image.random_crop(x_orig, tf.concat((tf.shape(x_orig)[:1], [20, 20], [n_chans]), axis=0))
        x = tf.stack([x1, x2])
        
        x = tf.transpose(x, [1, 0, 2, 3, 4])
        i = tf.squeeze(tf.random.categorical([[1., 1.]], tf.shape(x)[0]))
        x = x[0, i, ...]
        x = tf.image.resize(x, height_width)

    # testing transformations
    else:
        x = tf.image.central_crop(x_orig, np.mean(20 / np.array(x_orig.shape.as_list()[1:-1])))
        x = tf.image.resize(x, height_width)[0]

    return x



def get_mnist_test_x_aug_fn(mdl_input_dims = (28, 28, 1)):
    def test_fn(x_orig):
        return mnist_x_aug_fn(x_orig, mdl_input_dims, False)
    
    return test_fn



def mnist_gx_aug_fn(x_orig, mdl_input_dims = (24, 24, 1)):
    x_orig = tf.expand_dims(x_orig, 0)
    sample_repeats = 1

    # repeat samples accordingly
    x_orig = tf.tile(x_orig, [sample_repeats] + [1] * len(x_orig.shape.as_list()[1:]))

    # get common shapes
    height_width = mdl_input_dims[:-1]
    n_chans = mdl_input_dims[-1]

    # random rotation
    rad = 2 * np.pi * 25 / 360
    x_rot = tf.contrib.image.rotate(x_orig, tf.random.uniform(shape=tf.shape(x_orig)[:1], minval=-rad, maxval=rad))
    gx = tf.stack([x_orig, x_rot])
    gx = tf.transpose(gx, [1, 0, 2, 3, 4])
    i = tf.squeeze(tf.random.categorical([[1., 1.]], tf.shape(gx)[0]))
    gx = gx[:1, i, ...]
    
    # random crops
    x1 = tf.image.random_crop(gx, tf.concat((tf.shape(x_orig)[:1], [16, 16], [n_chans]), axis=0))
    x2 = tf.image.random_crop(gx, tf.concat((tf.shape(x_orig)[:1], [20, 20], [n_chans]), axis=0))
    x3 = tf.image.random_crop(gx, tf.concat((tf.shape(x_orig)[:1], [24, 24], [n_chans]), axis=0))
    gx = tf.stack([tf.image.resize(x1, height_width),
                   tf.image.resize(x2, height_width),
                   tf.image.resize(x3, height_width)])
    gx = tf.transpose(gx, [1, 0, 2, 3, 4])
    i = tf.squeeze(tf.random.categorical([[1., 1., 1.]], tf.shape(gx)[0]))
    gx = gx[:1, i, ...]

    # apply random adjustments
    def rand_adjust(img):
        img = tf.image.random_brightness(img, 0.4)
        img = tf.image.random_contrast(img, 0.6, 1.4)
        if img.shape.as_list()[-1] == 3:
            img = tf.image.random_saturation(img, 0.6, 1.4)
            img = tf.image.random_hue(img, 0.125)
        return img

    gx = tf.map_fn(lambda y: rand_adjust(y), gx, dtype=tf.float32)
    return gx[0]