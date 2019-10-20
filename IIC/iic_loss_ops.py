import tensorflow as tf

K = tf.keras.backend


def iic_loss(
    p_out: tf.Tensor,
    p_tf_out: tf.Tensor,
    lamb: float = 1.0,
    eps: float = 1e-6,
    name: str = None,
) -> tf.Tensor:
    """

    Args:
        p_out:
        p_tf_out:
        lamb:
        eps:

    Returns:

    """

    k = p_out.shape.as_list()[1]
    p_i_j = compute_joint(p_out, p_tf_out)

    assert len(p_out.shape.as_list()) == 2
    assert p_i_j.shape.as_list() == [k, k]

    # clip values
    p_i_j = tf.clip_by_value(p_i_j, eps, 1.0 / eps)

    # get marginals
    p_i = tf.broadcast_to(tf.reshape(tf.reduce_sum(p_i_j, axis=0), (k, 1)), (k, k))
    p_j = tf.broadcast_to(tf.reshape(tf.reduce_sum(p_i_j, axis=1), (1, k)), (k, k))

    loss = -p_i_j * (
        tf.math.log(p_i_j) - lamb * tf.math.log(p_j) - lamb * tf.math.log(p_i)
    )
    loss = tf.reduce_sum(loss)
    if name is None:
        name = "iic_loss"
    else:
        name = f"{name}/iic_loss"

    return tf.identity(loss, name=name)


def compute_joint(p_out: tf.Tensor, p_tf_out: tf.Tensor) -> tf.Tensor:
    """Compute joint probability matrix from paired tensors.
    p_out, p_tf_out should be probability vectors obtained from
    softmax activations.

    Args:
        p_out: [batch_size, num_classes]
        p_tf_out: [batch_size, num_classes]

    Returns:

    """
    assert p_out.shape.as_list() == p_tf_out.shape.as_list()

    p_out = tf.expand_dims(p_out, [-1])  # [bs, nc, 1]
    p_tf_out = tf.expand_dims(p_tf_out, [1])  # [bs, 1, nc]
    p_i_j = tf.matmul(p_out, p_tf_out)

    p_i_j = tf.reduce_sum(p_i_j, axis=0)  # [k, k]
    p_i_j = (p_i_j + tf.transpose(p_i_j)) / 2.0  # symmetrise
    p_i_j = p_i_j / tf.reduce_sum(p_i_j)  # normalise

    return p_i_j
