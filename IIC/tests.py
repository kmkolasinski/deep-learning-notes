#!/usr/bin/env python
# coding: utf-8

# In[5]:


get_ipython().system(
    "pip install image-classifiers tensorflow-datasets tqdm ipywidgets --upgrade"
)

# In[1]:


from classification_models.tfkeras import Classifiers
import tensorflow as tf
import tensorflow_datasets as tfds

keras = tf.keras
K = tf.keras.backend

# In[3]:


train_dataset, dataset_info = tfds.load(
    name="fashion_mnist", split=tfds.Split.TRAIN, as_supervised=True, with_info=True
)
test_dataset = tfds.load(
    name="fashion_mnist", split=tfds.Split.TEST, as_supervised=True
)

# In[4]:


dataset_info


# In[5]:


def normalize_labels(f, l):
    return tf.cast(f, tf.float32) / 255.0, l


batch_size = 256
image_size = 28
n_classes = 10
num_train_steps = 60000 // batch_size
num_eval_steps = 10000 // batch_size
train_dataset = (
    train_dataset.shuffle(1000).repeat(-1).batch(batch_size).map(normalize_labels)
)
test_dataset = test_dataset.batch(batch_size).map(normalize_labels)
train_dataset

# In[6]:


num_train_steps

# In[47]:


fl_iterator = test_dataset.make_one_shot_iterator().get_next()

# In[48]:


with tf.Session() as sess:
    fl_np = sess.run(fl_iterator)

# In[49]:


fl_np[0].max()

# In[7]:


# In[8]:


from classification_models.models.resnet import *
from classification_models.models._common_blocks import ChannelSE

kwargs = dict(
    backend=tf.keras.backend,
    layers=tf.keras.layers,
    models=tf.keras.models,
    keras_utils=tf.keras.utils,
)

input_shape = (image_size, image_size, 1)
params = ModelParams("custom", (1, 1, 1), residual_conv_block, ChannelSE)
base_model = ResNet(
    model_params=params,
    input_shape=input_shape,
    weights=None,
    include_top=False,
    **kwargs
)

# In[9]:


base_model.summary()

# In[10]:


inputs = keras.Input(shape=(image_size, image_size, 1), name="images")
feature_map = base_model(inputs)
x = keras.layers.GlobalAveragePooling2D()(feature_map)
output = keras.layers.Dense(n_classes, activation="softmax")(x)
cls_model = keras.models.Model(inputs=[inputs], outputs=[output])

# In[11]:


cls_model.summary()

# In[ ]:


# In[12]:


optimizer = keras.optimizers.Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.995)
cls_model.compile(
    optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# In[13]:


cls_model.fit(
    train_dataset,
    verbose=1,
    epochs=10,
    validation_data=test_dataset,
    validation_steps=num_eval_steps,
    steps_per_epoch=num_train_steps,
)

# In[14]:


optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.995)
cls_model.compile(
    optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# In[15]:


cls_model.fit(
    train_dataset,
    verbose=1,
    epochs=10,
    validation_data=test_dataset,
    validation_steps=num_eval_steps,
    steps_per_epoch=num_train_steps,
)

# In[16]:


get_ipython().system("pip install albumentations")

# In[156]:


from albumentations import (
    HorizontalFlip,
    IAAPerspective,
    ShiftScaleRotate,
    CLAHE,
    RandomRotate90,
    Transpose,
    ShiftScaleRotate,
    Blur,
    OpticalDistortion,
    GridDistortion,
    HueSaturationValue,
    IAAAdditiveGaussianNoise,
    GaussNoise,
    MotionBlur,
    MedianBlur,
    IAAPiecewiseAffine,
    IAASharpen,
    IAAEmboss,
    RandomBrightnessContrast,
    Flip,
    OneOf,
    Compose,
)
import numpy as np


def strong_aug(p=0.5):
    return Compose(
        [
            #         RandomRotate90(),
            #         Flip(),
            #         Transpose(),
            #         OneOf([
            #             IAAAdditiveGaussianNoise(),
            #             GaussNoise(),
            #         ], p=0.2),
            #         OneOf([
            #             MotionBlur(p=0.2),
            #             MedianBlur(blur_limit=3, p=0.1),
            #             Blur(blur_limit=3, p=0.1),
            #         ], p=0.2),
            ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=10, p=1.0),
            #         OneOf([
            #             OpticalDistortion(p=0.3),
            #             GridDistortion(p=0.1),
            #             IAAPiecewiseAffine(p=0.3),
            #         ], p=0.2),
            #         OneOf([
            #             CLAHE(clip_limit=2),
            #             IAASharpen(),
            #             IAAEmboss(),
            #             RandomBrightnessContrast(),
            #         ], p=0.3),
            #         HueSaturationValue(p=0.3),
        ],
        p=p,
    )


# In[157]:


train_dataset, dataset_info = tfds.load(
    name="fashion_mnist", split=tfds.Split.TRAIN, as_supervised=True, with_info=True
)

# In[158]:


augmentation_fn = strong_aug(p=0.5)


def tf_augmentation_fn(image):
    image = augmentation_fn(image=image)["image"]
    return image


# In[159]:


def normalize_labels_and_augment(f, l):
    image = tf.broadcast_to(f, (28, 28, 3))
    image = tf.numpy_function(tf_augmentation_fn, [image], tf.uint8)
    image = tf.cast(image[:, :, :1], tf.float32) / 255.0
    image.set_shape((28, 28, 1))
    return image, l


# In[160]:


train_dataset, dataset_info = tfds.load(
    name="fashion_mnist", split=tfds.Split.TRAIN, as_supervised=True, with_info=True
)
test_dataset = tfds.load(
    name="fashion_mnist", split=tfds.Split.TEST, as_supervised=True
)

train_dataset = (
    train_dataset.shuffle(1000)
    .repeat(-1)
    .map(normalize_labels_and_augment)
    .batch(batch_size)
    .prefetch(10)
)
test_dataset = test_dataset.batch(batch_size).map(normalize_labels)

# In[161]:


test_dataset, train_dataset

# In[162]:


train_dataset_iterator = train_dataset.make_one_shot_iterator().get_next()
with tf.Session() as sess:
    batch = sess.run(train_dataset_iterator)

# In[163]:


import matplotlib.pyplot as plt

# In[164]:


plt.imshow(batch[0][0, ..., 0])

# In[165]:


cls_model = keras.models.Model(inputs=[inputs], outputs=[output])
l2_loss = K.sum([keras.regularizers.l2(0.00001)(w) for w in cls_model.weights])
cls_model.add_loss(lambda: l2_loss)

optimizer = keras.optimizers.Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.995)
cls_model.compile(
    optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# In[166]:


cls_model.fit(
    train_dataset,
    verbose=1,
    epochs=5,
    validation_data=test_dataset,
    validation_steps=num_eval_steps,
    steps_per_epoch=num_train_steps,
)

# In[167]:


optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.995)
cls_model.compile(
    optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# In[168]:


cls_model.fit(
    train_dataset,
    verbose=1,
    epochs=5,
    validation_data=test_dataset,
    validation_steps=num_eval_steps,
    steps_per_epoch=num_train_steps,
)

# In[207]:


train_dataset, dataset_info = tfds.load(
    name="fashion_mnist", split=tfds.Split.TRAIN, as_supervised=True, with_info=True
)
test_dataset = tfds.load(
    name="fashion_mnist", split=tfds.Split.TEST, as_supervised=True
)


def strong_aug(p=0.5):
    return Compose(
        [
            #         RandomRotate90(),
            #         Flip(),
            #         Transpose(),
            OneOf([IAAAdditiveGaussianNoise(), GaussNoise()], p=0.5),
            #         OneOf([
            #             MotionBlur(p=0.2),
            #             MedianBlur(blur_limit=3, p=0.1),
            #             Blur(blur_limit=3, p=0.1),
            #         ], p=0.2),
            ShiftScaleRotate(
                shift_limit=0.2, scale_limit=0.1, rotate_limit=45, p=1.0, border_mode=1
            ),
            #         OneOf([
            #             OpticalDistortion(p=0.3),
            #             GridDistortion(p=0.1),
            #             IAAPiecewiseAffine(p=0.3),
            #         ], p=0.2),
            #         OneOf([
            #             CLAHE(clip_limit=2),
            #             IAASharpen(),
            #             IAAEmboss(),
            #             RandomBrightnessContrast(),
            #         ], p=0.3),
            #         HueSaturationValue(p=0.3),
        ],
        p=p,
    )


augmentation_fn = strong_aug(p=1.0)


def tf_augmentation_fn(image):
    image = augmentation_fn(image=image)["image"]
    return image


def normalize_labels_and_augment_v2(f, l):
    source_image = tf.cast(f, tf.float32) / 255.0
    image = tf.broadcast_to(f, (28, 28, 3))
    image = tf.numpy_function(tf_augmentation_fn, [image], tf.uint8)
    image = tf.cast(image[:, :, :1], tf.float32) / 255.0
    image.set_shape((28, 28, 1))
    return {"image": source_image, "tf_image": image}, {"label": l}


iic_train_dataset = (
    train_dataset.shuffle(1000)
    .repeat(-1)
    .map(normalize_labels_and_augment_v2)
    .batch(batch_size)
    .prefetch(10)
)
iic_test_dataset = (
    test_dataset.map(normalize_labels_and_augment_v2).batch(batch_size).prefetch(10)
)

# In[208]:


iic_train_dataset

# In[209]:


train_dataset_iterator = iic_train_dataset.make_one_shot_iterator().get_next()
with tf.Session() as sess:
    batch = sess.run(train_dataset_iterator)

# In[220]:


i = 19
plt.subplot(121)
plt.imshow(batch[0]["image"][i, ..., 0])
plt.subplot(122)
plt.imshow(batch[0]["tf_image"][i, ..., 0])

# In[230]:


image_input = keras.Input(shape=(image_size, image_size, 1), name="image")
tf_image_input = keras.Input(shape=(image_size, image_size, 1), name="tf_image")

cls_dense_layer = keras.layers.Dense(n_classes, activation="softmax")

feature_map = base_model(image_input)
x = keras.layers.GlobalAveragePooling2D()(feature_map)
image_output = cls_dense_layer(x)

feature_map = base_model(tf_image_input)
x = keras.layers.GlobalAveragePooling2D()(feature_map)
tf_image_output = cls_dense_layer(x)


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


distance = tf.keras.layers.Lambda(euclidean_distance, name="distance")(
    [image_output, tf_image_output]
)

cls_model = keras.models.Model(
    inputs={"image": image_input, "tf_image": tf_image_input},
    outputs={
        "image_output": image_output,
        "tf_image_output": tf_image_output,
        "distance": distance,
    },
)

l2_loss = K.sum([keras.regularizers.l2(0.00001)(w) for w in cls_model.weights])
cls_model.add_loss(lambda: l2_loss)

# In[231]:


cls_model.outputs

# In[ ]:


# In[232]:


optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.995)
cls_model.compile(optimizer=optimizer, loss={"distance": "mse"}, metrics=[])

# In[235]:


cls_model.fit(
    iic_train_dataset,
    verbose=1,
    epochs=1,
    #     validation_data=test_dataset,
    #     validation_steps=num_eval_steps,
    steps_per_epoch=num_train_steps,
)

# In[ ]:
