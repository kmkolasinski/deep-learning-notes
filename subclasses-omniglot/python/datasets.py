import pickle

import numpy as np
import tensorflow as tf
from tensorflow import keras
from natsort import natsorted
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm


prepro = keras.preprocessing.image


def build_dataset(images_list, image_size, out_name='omniglot_training'):
    omni_dataset = []
    classes_level0 = []
    classes_level1 = []
    for path in tqdm(images_list):
        alphabet = path.split('/')[-3]
        character = path.split('/')[-2]
        image = resize(imread(path), image_size, mode='constant')

        class_name = alphabet + "/" + character
        omni_dataset.append(image)
        classes_level0.append(alphabet)
        classes_level1.append(class_name)

    classes_level0_mapping = {k: i for i, k in
                              enumerate(natsorted(set(classes_level0)))}
    classes_level1_mapping = {k: i for i, k in
                              enumerate(natsorted(set(classes_level1)))}

    labels_level0 = [classes_level0_mapping[c] for c in classes_level0]
    labels_level1 = [classes_level1_mapping[c] for c in classes_level1]

    training_data = {
        'image': np.expand_dims(np.array(omni_dataset), -1).astype(np.float32),
        'label0': keras.utils.to_categorical(labels_level0).astype(np.float32),
        'label1': keras.utils.to_categorical(labels_level1).astype(np.float32)
    }

    pickle.dump(training_data, open(f"../data/{out_name}.pkl", 'wb'))
    pickle.dump(classes_level0_mapping,
                open("../data/classes_level0_mapping.pkl", 'wb'))
    pickle.dump(classes_level1_mapping,
                open("../data/classes_level1_mapping.pkl", 'wb'))


def image_augmentation(image, label0, label1):
    params = dict(row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest',
                  cval=1.0)

    image = prepro.random_rotation(image, rg=10, **params)
    image = prepro.random_shift(image, wrg=0.1, hrg=0.1, **params)
    image = prepro.random_shear(image, intensity=0.1, **params)
    image = prepro.random_zoom(image, zoom_range=(0.8, 1.2), **params)

    return image, label0, label1


def no_image_augmentation(image, label0, label1):
    return image, label0, label1


def select_labels0(image, label0, label1):
    return {'image': image}, {'label': label0}


def input_function(data: np.ndarray, batch_size, do_augmentation=True):
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.shuffle(buffer_size=10000)
    if do_augmentation:
        dataset = dataset.map(
            lambda example: tuple(tf.py_func(
                image_augmentation,
                [example['image'], example['label0'], example['label1']],
                [tf.float32, tf.float32, tf.float32])), num_parallel_calls=16
        )
    else:
        dataset = dataset.map(
            lambda example: tuple(tf.py_func(
                no_image_augmentation,
                [example['image'], example['label0'], example['label1']],
                [tf.float32, tf.float32, tf.float32])), num_parallel_calls=16
        )

    dataset = dataset.repeat(-1)
    dataset = dataset.map(select_labels0, num_parallel_calls=16)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2)

    iterator = dataset.make_one_shot_iterator()
    batch_generator = iterator.get_next()

    return batch_generator


def get_input_function(dataset_path: str, batch_size: int, do_augmentation: bool):
    data = pickle.load(open(dataset_path, 'rb'))
    num_examples = data['image'].shape[0]

    def func():
        return input_function(data, batch_size, do_augmentation)

    return func, num_examples
