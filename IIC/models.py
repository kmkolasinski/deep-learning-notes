from typing import Tuple

from classification_models.models.resnet import *
from classification_models.models._common_blocks import ChannelSE
import tensorflow as tf

import iic_loss_ops

keras = tf.keras
K = tf.keras.backend

keras_kwargs = dict(
    backend=tf.keras.backend,
    layers=tf.keras.layers,
    models=tf.keras.models,
    keras_utils=tf.keras.utils,
)


def create_resnet_se_backbone(
    input_shape: Tuple[int, int, int], units_per_block: Tuple[int, ...] = (2, 2)
):

    params = ModelParams(
        "CustomResnetSE", units_per_block, residual_conv_block, ChannelSE
    )
    base_model = ResNet(
        model_params=params,
        input_shape=input_shape,
        weights=None,
        include_top=False,
        **keras_kwargs,
    )
    return base_model


def create_iic_model(
    input_shape: Tuple[int, int, int],
    base_model: keras.Model,
    main_head_num_classes: int,
    aux_head_num_classes: int = None,
    num_main_heads: int = 1,
    num_aux_heads: int = 0,
):
    image_input = keras.Input(shape=input_shape, name="image")
    tf_image_input = keras.Input(shape=input_shape, name="tf_image")
    inputs = {"image": image_input, "tf_image": tf_image_input}

    def project(head, h, name):
        h = keras.layers.GlobalAveragePooling2D()(base_model(h))
        return keras.layers.Lambda(lambda x: x, name=name)(head(h))

    main_head_outputs = []
    for i in range(num_main_heads):
        head_name = f"main_head_{i}"
        head_layer = keras.layers.Dense(main_head_num_classes, activation="softmax")

        p_out = project(head_layer, image_input, f"{head_name}/p_out")
        p_tf_out = project(head_layer, tf_image_input, f"{head_name}/p_tf_out")
        iic_loss = iic_loss_ops.iic_loss(p_out=p_out, p_tf_out=p_tf_out, name=head_name)

        main_head_outputs.append(dict(outputs=[p_out, p_tf_out], loss=iic_loss))

    aux_head_outputs = []
    for i in range(num_aux_heads):
        head_name = f"aux_head_{i}"
        assert aux_head_num_classes is not None and aux_head_num_classes > 0
        head_layer = keras.layers.Dense(aux_head_num_classes, activation="softmax")

        p_out = project(head_layer, image_input, f"{head_name}/p_out")
        p_tf_out = project(head_layer, tf_image_input, f"{head_name}/p_tf_out")
        iic_loss = iic_loss_ops.iic_loss(p_out=p_out, p_tf_out=p_tf_out, name=head_name)

        aux_head_outputs.append(dict(outputs=[p_out, p_tf_out], loss=iic_loss))

    return inputs, main_head_outputs, aux_head_outputs
