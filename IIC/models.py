from typing import Tuple, List, Optional, Dict

import tensorflow as tf
from classification_models.models.resnet import *

from iic_loss_ops import iic_loss

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


def get_iic_target_loss(
    model_outputs: Dict[str, List[tf.Tensor]],
    model_tf_outputs: Dict[str, List[tf.Tensor]],
):
    mean_head_losses = {}
    sub_heads_losses = {}
    for head_name, heads_p_out in model_outputs.items():
        if len(heads_p_out) == 0:
            continue
        num_heads = len(heads_p_out)
        heads_p_tf_out = model_tf_outputs[head_name]
        sub_head_losses = [
            iic_loss(heads_p_out[k], heads_p_tf_out[k]) for k in range(num_heads)
        ]
        mean_head_losses[f"{head_name}/mean_loss"] = tf.add_n(sub_head_losses) / num_heads
        sub_heads_losses[head_name] = {f"{head_name}/head_{k}_loss": l for k, l in enumerate(sub_head_losses)}

    return mean_head_losses, sub_heads_losses


def add_losses_to_model(model: keras.Model, losses: Dict[str, tf.Tensor]) -> None:
    for name, loss in losses.items():
        print(f"Adding loss to model: {name}")
        model.add_loss(loss)
    add_metrics_to_model(model, losses)


def add_metrics_to_model(model: keras.Model, metrics: Dict[str, tf.Tensor]) -> None:
    for name, metric_value in metrics.items():
        model.add_metric(metric_value, name=name, aggregation='mean')
