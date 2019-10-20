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
        image, tf.random_uniform(shape=[], minval=-math.pi / 4, maxval=math.pi / 4)
    )
    alpha = tf.random_uniform(shape=[], minval=-0.5, maxval=0.5)
    transform = [1, tf.sin(alpha), 0, 0, tf.cos(alpha), 0, 0, 0]
    image = tf.contrib.image.transform(image, transform, interpolation="NEAREST")
    image = tf.contrib.image.translate(
        image, tf.random_uniform(shape=[2], minval=-height / 8, maxval=height / 8)
    )

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
    noise = tf.random_uniform(tf.shape(image), minval=-0.05, maxval=0.05)
    image = image + noise

    return image


def prepare_image_pairs(aug_fn, num_repeats: int = 2):
    def create_image_pair(features, labels):
        image, label = features["image"], labels["label"]
        images, labels = [image] * num_repeats, [label] * num_repeats
        tf_images = []
        for _ in range(num_repeats):
            tf_images.append(aug_fn(image))

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
