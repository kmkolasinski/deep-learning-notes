import math
from collections import defaultdict
from typing import Tuple
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tqdm import tqdm


def default_preprocess_features_and_labels(image: tf.Tensor, label: tf.Tensor):
    """Default image preprocessing function compatible with MNIST dataset"""
    image = tf.cast(image, tf.float32) / 255.0
    return {"image": image}, {"label": label}


def apply_sobel(image: tf.Tensor, method: str = "sobel") -> tf.Tensor:

    sobel1 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    sobel = (
        np.array([sobel1, sobel2])
        .transpose([1, 2, 0])
        .reshape([3, 3, 1, 2])
        .astype(np.float32)
    )

    gray_image = tf.reduce_mean(image, axis=-1, keepdims=True)
    gray_image = tf.expand_dims(gray_image, 0)
    sobel_image = tf.nn.conv2d(gray_image, sobel, padding="SAME")

    if method == "sobel":
        return sobel_image[0]
    if method == "sobel_gray":
        return tf.concat([sobel_image, gray_image], axis=-1)[0]
    if method == "sobel_rgb":
        return tf.concat([sobel_image, tf.expand_dims(image, 0)], axis=-1)[0]

    raise ValueError("Not implemented sobel method")


def get_default_image_aug_fn(
    rotation: float = 0.15,
    skew: float = 0.1,
    translation: float = 0.15,
    crop: float = 0.3,
    center_crop: float = 0.0,
    hue: float = 0.05,
    saturation: float = 0.3,
    brightness: float = 0.3,
    contrast: float = 0.3,
    uniform_noise: float = 0.02,
    sobel: str = None,
):
    center_crop_fraction = 1.0 - center_crop

    def aug_fn(image: tf.Tensor) -> tf.Tensor:

        if sobel is not None:
            image = apply_sobel(image, sobel)

        height, width, num_channels = image.shape.as_list()

        if rotation != 0:
            alpha = tf.random.uniform(
                shape=[], minval=-math.pi * rotation, maxval=math.pi * rotation
            )
            image = tfa.image.rotate(image, alpha)

        if skew != 0:
            alpha = tf.random.uniform(shape=[], minval=-skew, maxval=skew)
            transform = [1, tf.sin(alpha), 0, 0, tf.cos(alpha), 0, 0, 0]
            image = tfa.image.transform(image, transform, interpolation="BILINEAR")

        if translation != 0:
            random_ty = tf.random.uniform(
                shape=[], minval=-height * translation, maxval=height * translation
            )
            random_tx = tf.random.uniform(
                shape=[], minval=-width * translation, maxval=width * translation
            )
            random_translation = tf.stack([random_ty, random_tx])
            image = tfa.image.translate(image, random_translation)

        # random cropping
        if crop != 0:
            min_height = int(height * (1 - crop))
            min_width = int(width * (1 - crop))

            random_crop_height = tf.random.uniform(
                shape=[], minval=min_height, maxval=height, dtype=tf.int32
            )
            random_crop_width = tf.random.uniform(
                shape=[], minval=min_width, maxval=width, dtype=tf.int32
            )
            random_crop_size = tf.stack(
                [random_crop_height, random_crop_width, num_channels]
            )

            image = tf.image.random_crop(image, random_crop_size)
            image = tf.image.resize(image, (height, width))

        if center_crop != 0:
            image = tf.image.central_crop(image, center_crop_fraction)
            image = tf.image.resize(image, (height, width))

        # additional processing for RGB images
        if num_channels == 3:
            if hue != 0:
                image = tf.image.random_hue(image, hue)
            if saturation != 0:
                image = tf.image.random_saturation(
                    image, 1 - saturation, 1 + saturation
                )
            if brightness != 0:
                image = tf.image.random_brightness(image, brightness)
            if contrast != 0:
                image = tf.image.random_contrast(image, 1 - contrast, 1 + contrast)

        # add random noise
        if uniform_noise != 0:
            amplitude = tf.random.uniform([], minval=0.0, maxval=1.0)
            noise = tf.random.uniform(
                tf.shape(image), minval=-uniform_noise, maxval=uniform_noise
            )
            image = image + amplitude * noise

        image = tf.clip_by_value(image, 0.0, 1.0)
        return image

    return aug_fn


def prepare_image_pairs(
    first_image_aug_fn, second_image_aug_fn=None, num_repeats: int = 2
):
    def create_image_pair(features, labels):
        image, label = features["image"], labels["label"]
        images, labels = [], [label] * num_repeats
        tf_images = []

        for _ in range(num_repeats):
            images.append(first_image_aug_fn(image))
            if second_image_aug_fn is not None:
                tf_images.append(second_image_aug_fn(image))

        features = {"image": tf.stack(images)}
        labels = {"label": tf.stack(labels)}

        if second_image_aug_fn is not None:
            features["tf_image"] = tf.stack(tf_images)

        return features, labels

    return create_image_pair


def fix_batch_dimension(features, labels, input_size=None):
    def resize(image: tf.Tensor) -> tf.Tensor:
        height, width, num_channels = image.shape.as_list()[2:]
        batch_shape = [-1, height, width, num_channels]
        image = tf.reshape(image, batch_shape)
        if input_size is not None:
            image = tf.image.resize(image, input_size)
        return image

    new_features = {k: resize(image) for k, image in features.items()}
    new_labels = {"label": tf.reshape(labels["label"], [-1])}
    return new_features, new_labels


def prepare_dataset(
    dataset: tf.data.Dataset,
    dataset_preprocess_fn=default_preprocess_features_and_labels,
    first_image_aug_fn=get_default_image_aug_fn(),
    second_image_aug_fn=get_default_image_aug_fn(),
    input_size: Tuple[int, int] = None,
    batch_size: int = 64,
    num_image_repeats: int = 1,
    dataset_repeats: int = -1,
    shuffle_size: int = 4096,
    batch_prefetch_size: int = 10,
) -> Tuple[tf.data.Dataset, Tuple[int, int, int]]:

    autotune = tf.data.experimental.AUTOTUNE
    pair_fn = prepare_image_pairs(
        first_image_aug_fn, second_image_aug_fn, num_image_repeats
    )

    def set_dimensions(features, labels):
        return fix_batch_dimension(features, labels, input_size=input_size)

    dataset = (
        dataset.repeat(dataset_repeats)
        .shuffle(shuffle_size)
        .map(dataset_preprocess_fn, autotune)
        .map(pair_fn, autotune)
        .batch(batch_size)
        .map(set_dimensions, autotune)
        .prefetch(batch_prefetch_size)
    )

    model_input_shape = dataset.output_shapes[0]["image"].as_list()[1:]

    return dataset, model_input_shape


def image_per_pixel_euclidean_distance_fn(
    image: np.ndarray, images: np.ndarray
) -> np.ndarray:
    num_images = images.shape[0]
    image = image.reshape([1, -1]) / 255.0
    images = images.reshape([num_images, -1]) / 255.0
    return np.sqrt((image - images) ** 2).mean(-1)


def select_most_distance_vectors(
    vectors: np.ndarray, k: int, distance_fn=image_per_pixel_euclidean_distance_fn
) -> np.ndarray:
    """

    Args:
        vectors: array of shape [num_examples, ...]
        k: integer k < num_examples
        distance_fn:

    Returns:

    """
    num_images = vectors.shape[0]
    farthest_pts = np.zeros((k, *vectors.shape[1:]), dtype=np.uint8)
    farthest_pts[0] = vectors[np.random.randint(len(vectors))]
    distances = distance_fn(farthest_pts[0], vectors)
    assert distances.shape == (num_images,)

    for i in range(1, k):
        farthest_pts[i] = vectors[np.argmax(distances)]
        distances = np.minimum(distances, distance_fn(farthest_pts[i], vectors))
    return farthest_pts


def sample_classification_dataset(
    dataset: tf.data.Dataset,
    min_num_occurrences: int = 20,
    class_buffer_size: int = 100,
    reject_rare_classes: bool = True,
    distance_fn=image_per_pixel_euclidean_distance_fn,
    as_numpy: bool = False,
):

    assert dataset.output_types == (tf.uint8, tf.int64)
    # expecting image of fixed size
    assert len(dataset.output_shapes[0]) == 3
    assert all([c is not None for c in dataset.output_shapes[0]])
    assert dataset.output_shapes[1] == []

    dataset_iterator = dataset.repeat(1).make_one_shot_iterator()

    label_images = defaultdict(list)
    for image, label in tqdm(dataset_iterator):
        label = label.numpy()
        image = image.numpy()
        if len(label_images[label]) < class_buffer_size:
            label_images[label].append(image)

    print(f"Found n={len(label_images)} classes in dataset.")
    if reject_rare_classes:
        label_images = {
            l: im for l, im in label_images.items() if len(im) >= min_num_occurrences
        }
        print(f"Number of classes after rejections is {len(label_images)}.")

    print(f"Selecting most distance examples from buffers ...")
    selected_images = []
    selected_labels = []
    for label, images in tqdm(label_images.items()):
        images = np.array(images)
        images = select_most_distance_vectors(
            images, min_num_occurrences, distance_fn=distance_fn
        )
        labels = np.array([label] * min_num_occurrences)
        selected_images.append(images)
        selected_labels.append(labels)
    images = np.vstack(selected_images)
    labels = np.hstack(selected_labels)
    if as_numpy:
        return images, labels

    return tf.data.Dataset.from_tensor_slices((images, labels))
