import math
import tensorflow as tf


def default_preprocess_features_and_labels(image: tf.Tensor, label: tf.Tensor):
    """Default image preprocessing function compatible with MNIST dataset"""
    image = tf.cast(image, tf.float32) / 255.0
    return {"image": image}, {"label": label}


def default_image_augmentation(image: tf.Tensor) -> tf.Tensor:

    height, width, num_channels = image.shape.as_list()

    # random transformations
    image = tf.contrib.image.rotate(
        image, tf.random_uniform(shape=[], minval=-math.pi / 7, maxval=math.pi / 7)
    )
    
#     alpha = tf.random_uniform(shape=[], minval=-0.5, maxval=0.5)
#     transform = [1, tf.sin(alpha), 0, 0, tf.cos(alpha), 0, 0, 0]
#     image = tf.contrib.image.transform(image, transform, interpolation="NEAREST")
    
#     image = tf.contrib.image.translate(
#         image, tf.random_uniform(shape=[2], minval=-height / 8, maxval=height / 8)
#     )

    # random cropping
    min_height = int(height * 0.8)
    min_width = int(width * 0.8)
    random_crop_height = tf.random_uniform(
        shape=[], minval=min_height, maxval=height, dtype=tf.int32
    )
    random_crop_width = tf.random_uniform(
        shape=[], minval=min_width, maxval=width, dtype=tf.int32
    )
    random_crop_size = tf.stack([random_crop_height, random_crop_width, num_channels])
    image = tf.image.random_crop(image, random_crop_size)
    image = tf.image.resize(image, (height, width))

    # additional processing for RGB images
    if num_channels == 3:
        image = tf.image.random_hue(image, 0.08)
        image = tf.image.random_saturation(image, 0.6, 1.6)
        image = tf.image.random_brightness(image, 0.05)
        image = tf.image.random_contrast(image, 0.7, 1.3)

    # add random noise
#     noise = tf.random_uniform(tf.shape(image), minval=-0.05, maxval=0.05)
#     image = image + noise

    return image


def prepare_image_pairs(aug_fn, num_repeats: int = 2):
    def create_image_pair(features, labels):
        image, label = features["image"], labels["label"]
        images, labels = [], [label] * num_repeats
        tf_images = []
        ims = [24, 24, 1]
        image = tf.expand_dims(image, 0)
        for _ in range(num_repeats):            
            images.append(mnist_x(image, (24, 24, 1), True))
            tf_images.append(mnist_gx(image, (24, 24, 1), True))            
#             tf_images.append(aug_fn(image))
        print(tf.stack(images))
        return (
            {"image": tf.stack(images), "tf_image": tf.stack(tf_images)},
            {"label": tf.stack(labels)},
        )

    return create_image_pair


def fix_batch_dimension(features, labels):
    height, width, num_channels = features["image"].shape.as_list()[2:]
    batch_shape = [-1, height, width, num_channels]
    return (
        {
            "image": tf.reshape(features["image"], batch_shape),
            "tf_image": tf.reshape(features["tf_image"], batch_shape),
        },
        {"label": tf.reshape(labels["label"], [-1])},
    )


def prepare_training_dataset(
    dataset: tf.data.Dataset,
    dataset_preprocess_fn=default_preprocess_features_and_labels,
    image_augmentation_fn=default_image_augmentation,
    batch_size: int = 64,
    num_image_repeats: int = 1,
    dataset_repeats: int = -1,
    shuffle_size: int = 1024,
    batch_prefetch_size: int = 10,
) -> tf.data.Dataset:

    autotune = tf.data.experimental.AUTOTUNE
    pair_fn = prepare_image_pairs(image_augmentation_fn, num_image_repeats)
    return (
        dataset.repeat(dataset_repeats)
        .shuffle(shuffle_size)
        .map(dataset_preprocess_fn, autotune)
        .map(pair_fn, autotune)
        .batch(batch_size)
        .map(fix_batch_dimension)
        .prefetch(batch_prefetch_size)
    )


import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def mnist_x(x_orig, mdl_input_dims, is_training):

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
        x = tf.image.resize(x, height_width)

    return x


def mnist_gx(x_orig, mdl_input_dims, is_training):
    sample_repeats = 1
    # if not training, return a constant value--it will unused but needs to be same shape to avoid TensorFlow errors
    if not is_training:
        return tf.zeros([0] + mdl_input_dims)

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
#     gx = tf.map_fn(lambda y: y[0][y[1]], (gx, i), dtype=tf.float32)

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
#     gx = tf.map_fn(lambda y: y[0][y[1]], (gx, i), dtype=tf.float32)

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