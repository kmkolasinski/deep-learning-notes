import tensorflow as tf


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += mask * -1e9

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(
        scaled_attention_logits, axis=-1
    )  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class Layer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(Layer, self).__init__(*args, **kwargs)

    def call(self, inputs, mask=None, **kwargs):
        if type(inputs) in [list, tuple]:
            return self._call(*inputs, **kwargs)
        elif type(inputs) == tf.Tensor:
            return self._call(inputs, **kwargs)
        else:
            raise NotImplementedError(
                f"Invalid input type to layer: {self.__class__.__name__}: {inputs}"
            )


class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def _call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask
        )

        scaled_attention = tf.transpose(
            scaled_attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.d_model)
        )  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        return output, attention_weights


class MLP(Layer):
    def __init__(self, depths, output_dim, activation="relu"):
        super(MLP, self).__init__()
        layers = []
        for depth in depths:
            layers += [
                tf.keras.layers.Dense(depth, activation=None),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation(activation),
            ]
        layers += [tf.keras.layers.Dense(output_dim, activation=None)]
        self.model = tf.keras.Sequential(layers)

    def _call(self, x, training = None, **kwargs):
        return self.model(x)


class KeypointEncoderLayer(Layer):
    def __init__(self, mlp_model: tf.keras.layers.Layer):
        super(KeypointEncoderLayer, self).__init__()
        self.ffn = mlp_model

    def _call(self, d, p, training):
        # (batch_size, input_seq_len, depth)
        x = d + self.ffn(p, training=training)
        return x


class AttentionalMessagePassingLayer(Layer):
    def __init__(self, d_model, num_heads, mle_model: tf.keras.layers.Layer):
        super(AttentionalMessagePassingLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = mle_model
        self.attention_weights = []

    def _call(self, xi, xj, mask, training):
        # (batch_size, input_seq_len, d_model)
        m_epsilon, attention_weights = self.mha([xj, xj, xi, mask])
        self.attention_weights.append((xi, xj, attention_weights))
        # (batch_size, input_seq_len, 2 * d_model)
        output = tf.concat([xi, m_epsilon], axis=-1)
        # (batch_size, input_seq_len, d_model)
        ffn_output = self.ffn(output, training=training)
        return xi + ffn_output


class SuperGlue(Layer):
    def __init__(
        self,
        depth: int,
        num_layers: int,
        num_heads: int,
        keypoint_encoder=None,
        layer_mlp_fn=None,
    ):
        super(SuperGlue, self).__init__()
        if layer_mlp_fn is None:
            layer_mlp_fn = lambda: MLP([2 * depth], depth)

        if keypoint_encoder is None:
            self.keypoint_encoder = KeypointEncoderLayer(MLP([32, 64, 128, 256], depth))
        else:
            self.keypoint_encoder = keypoint_encoder

        self.amp_layers = []
        for i in range(num_layers * 2):
            amp = AttentionalMessagePassingLayer(depth, num_heads, layer_mlp_fn())
            self.amp_layers.append(amp)

        self.projection_layer = tf.keras.layers.Dense(depth, activation=None)

    def _call(self, dR, pR, dL, pL, training):

        xR = self.keypoint_encoder([dR, pR], training=training)
        xL = self.keypoint_encoder([dL, pL], training=training)

        for i, amp_layer in enumerate(self.amp_layers):
            l = i + 1
            if l % 2 == 1:
                # self attention
                xR = amp_layer([xR, xR, None], training=training)
                xL = amp_layer([xL, xL, None], training=training)
            else:
                # cross attention
                xR_cross = amp_layer([xR, xL, None], training=training)
                xL_cross = amp_layer([xL, xR, None], training=training)
                xR = xR_cross
                xL = xL_cross

        fR = self.projection_layer(xR)
        fL = self.projection_layer(xL)
        return fR, fL
