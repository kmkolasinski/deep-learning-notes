from typing import Tuple, List, Optional

import tensorflow as tf
from classification_models.models.resnet import *

keras = tf.keras
K = tf.keras.backend

keras_kwargs = dict(
    backend=tf.keras.backend,
    layers=tf.keras.layers,
    models=tf.keras.models,
    keras_utils=tf.keras.utils,
)


def create_resnet_se_backbone(
    input_shape: Tuple[int, int, int],
    units_per_block: Tuple[int, ...] = (1, 1),
    attention=ChannelSE,
    name="CustomResnet",
):

    params = ModelParams(name, units_per_block, residual_conv_block, attention)
    base_model = ResNet(
        model_params=params,
        input_shape=input_shape,
        weights=None,
        include_top=False,
        **keras_kwargs,
    )
    return base_model


def create_head(hidden: tf.Tensor, head_size: int, name: str) -> tf.Tensor:
    head = keras.layers.Dense(head_size, activation="softmax", name=f"{name}/dense")
    p_out = keras.layers.Lambda(lambda x: x, name=f"{name}/p_out")(head(hidden))
    return p_out


def create_iic_model(
    base_model: keras.Model,
    main_heads_num_classes: List[int],
    aux_heads_num_classes: Optional[List[int]] = None,
):
    input_shape = base_model.input_shape[1:]
    image_input = keras.Input(shape=input_shape, name="image")

    hidden = keras.layers.GlobalAveragePooling2D()(base_model(image_input))

    if aux_heads_num_classes is None:
        aux_heads_num_classes = []

    main_head_outputs = []
    for i, nc in enumerate(main_heads_num_classes):
        head_name = f"main_head_{i}"
        main_head_outputs.append(create_head(hidden, nc, head_name))

    aux_head_outputs = []
    for i, nc in enumerate(aux_heads_num_classes):
        head_name = f"aux_head_{i}"
        aux_head_outputs.append(create_head(hidden, nc, head_name))

    return keras.Model(
        image_input, {"main_heads": main_head_outputs, "aux_heads": aux_head_outputs}
    )
