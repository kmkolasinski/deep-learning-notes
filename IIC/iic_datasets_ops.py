import math
import tensorflow as tf


def default_preprocess_features_and_labels(image: tf.Tensor, label: tf.Tensor):
    """Default image preprocessing function compatible with MNIST dataset"""
    image = tf.cast(image, tf.float32) / 255.0
    return {"image": image}, {"label": label}


def get_default_image_aug_fn(
    rotation: float = 0.15, 
    skew: float = 0.1, 
    translation: float = 0.15,
    crop: float = 0.3,
    hue: float = 0.05,
    saturation: float = 0.2,
    brightness: float = 0.05,
    contrast: float = 0.3,
    uniform_noise: float = 0.01
):
    
    def aug_fn(image: tf.Tensor) -> tf.Tensor:

        height, width, num_channels = image.shape.as_list()
        
        if rotation != 0:
            alpha = tf.random_uniform(shape=[], minval=-math.pi * rotation, maxval=math.pi * rotation)
            image = tf.contrib.image.rotate(image, alpha)
        
        if skew != 0:
            alpha = tf.random_uniform(shape=[], minval=-skew, maxval=skew)
            transform = [1, tf.sin(alpha), 0, 0, tf.cos(alpha), 0, 0, 0]
            image = tf.contrib.image.transform(image, transform, interpolation="BILINEAR")
        
        if translation != 0:            
            random_ty = tf.random_uniform(shape=[], minval=-height * translation, maxval=height * translation)
            random_tx = tf.random_uniform(shape=[], minval=-width * translation, maxval=width * translation)
            random_translation = tf.stack([random_ty, random_tx])            
            image = tf.contrib.image.translate(image, random_translation)

        # random cropping
        if crop != 0:
            min_height = int(height * (1 - crop))
            min_width = int(width * (1 - crop))
            
            random_crop_height = tf.random_uniform(shape=[], minval=min_height, maxval=height, dtype=tf.int32)
            random_crop_width = tf.random_uniform(shape=[], minval=min_width, maxval=width, dtype=tf.int32)
            random_crop_size = tf.stack([random_crop_height, random_crop_width, num_channels])
            
            image = tf.image.random_crop(image, random_crop_size)
            image = tf.image.resize(image, (height, width))

        # additional processing for RGB images
        if num_channels == 3:
            if hue != 0:
                image = tf.image.random_hue(image, hue)
            if saturation != 0:
                image = tf.image.random_saturation(image, 1 - saturation, 1 + saturation)
            if brightness != 0:
                image = tf.image.random_brightness(image, brightness)
            if contrast != 0:
                image = tf.image.random_contrast(image, 1 - contrast, 1 + contrast)

        # add random noise
        if uniform_noise != 0:
            noise = tf.random_uniform(tf.shape(image), minval=-uniform_noise, maxval=uniform_noise)
            image = image + noise
            
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image
    return aug_fn


def prepare_image_pairs(first_image_aug_fn, second_image_aug_fn = None, num_repeats: int = 2):
    if second_image_aug_fn is None:
        second_image_aug_fn = first_image_aug_fn
        
    def create_image_pair(features, labels):
        image, label = features["image"], labels["label"]
        images, labels = [], [label] * num_repeats
        tf_images = []
        
        for _ in range(num_repeats):            
            images.append(first_image_aug_fn(image))
            tf_images.append(second_image_aug_fn(image))            

        outputs = (
            {"image": tf.stack(images), "tf_image": tf.stack(tf_images)},
            {"label": tf.stack(labels)},
        )
        print(outputs)
        return outputs
    return create_image_pair


def fix_batch_dimension(features, labels, input_size=None):
    
    def resize(image: tf.Tensor) -> tf.Tensor:
        height, width, num_channels = image.shape.as_list()[2:]
        batch_shape = [-1, height, width, num_channels]
        image = tf.reshape(image, batch_shape)
        if input_size is not None:
            image = tf.image.resize(image, input_size)
            
        return image
    
    return (
        {
            "image": resize(features["image"]),
            "tf_image": resize(features["tf_image"]),
        },
        {"label": tf.reshape(labels["label"], [-1])},
    )


def prepare_training_dataset(
    dataset: tf.data.Dataset,
    dataset_preprocess_fn=default_preprocess_features_and_labels,
    first_image_aug_fn=get_default_image_aug_fn(),
    second_image_aug_fn=get_default_image_aug_fn(),
    input_size = None,
    batch_size: int = 64,
    num_image_repeats: int = 1,
    dataset_repeats: int = -1,
    shuffle_size: int = 4096,
    batch_prefetch_size: int = 10,
) -> tf.data.Dataset:

    autotune = tf.data.experimental.AUTOTUNE
    pair_fn = prepare_image_pairs(first_image_aug_fn, second_image_aug_fn, num_image_repeats)
    
    def set_dimensions(features, labels):
        return fix_batch_dimension(features, labels, input_size=input_size)
    
    dataset = dataset.repeat(dataset_repeats) \
        .shuffle(shuffle_size) \
        .map(dataset_preprocess_fn, autotune) \
        .map(pair_fn, autotune) \
        .batch(batch_size) \
        .map(set_dimensions, autotune) \
        .prefetch(batch_prefetch_size)
    
    model_input_shape = dataset.output_shapes[0]['image'].as_list()[1:]
    
    return (dataset, model_input_shape)

