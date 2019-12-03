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


class MultiHeadAttention(tf.keras.layers.Layer):
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

    def call(self, v, k, q, mask):
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


class SuperGlueMLP(tf.keras.layers.Layer):
    def __init__(self, depths, activation="relu"):
        super(SuperGlueMLP, self).__init__()
        # TODO make if close to description in the paper
        assert len(depths) >= 1
        output_depth = depths[-1]
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(depth, activation=activation)
                for depth in depths[:-1]
            ]
            + [tf.keras.layers.Dense(output_depth)]  # (batch_size, seq_len, d_model)
        )

    def call(self, x, is_training, **kwargs):
        return self.model(x)


class KeypointEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, mle_model: tf.keras.layers.Layer):
        super(KeypointEncoderLayer, self).__init__()
        self.ffn = mle_model

    def call(self, d, p, is_training):
        x = d + self.ffn(
            p, is_training=is_training
        )  # (batch_size, input_seq_len, depth)
        return x


class AttentionalMessagePassingLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, mle_model: tf.keras.layers.Layer):
        super(AttentionalMessagePassingLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = mle_model

    def call(self, xi, xj, mask, is_training):
        # (batch_size, input_seq_len, d_model)
        m_epsilon, _ = self.mha(xj, xj, xi, mask)
        # (batch_size, input_seq_len, 2 * d_model)
        output = tf.concat([xi, m_epsilon], axis=-1)
        # (batch_size, input_seq_len, d_model)
        ffn_output = self.ffn(output, is_training=is_training)
        return xi + ffn_output
