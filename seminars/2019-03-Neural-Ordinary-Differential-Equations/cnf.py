import tensorflow as tf

keras = tf.keras


class HyperNet(tf.keras.Model):

    def __init__(self, input_dim, hidden_dim, n_ensemble):
        super().__init__()

        blocksize = n_ensemble * input_dim
        self._layers = [
            keras.layers.Dense(hidden_dim, activation="tanh"),
            keras.layers.Dense(hidden_dim, activation="tanh"),
            keras.layers.Dense(3 * blocksize + n_ensemble),
        ]
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_ensemble = n_ensemble
        self.blocksize = blocksize

    def call(self, t: tf.Tensor, **kwargs):
        t = tf.reshape(t, [1, 1])
        params = t
        for layer in self._layers:
            params = layer(params)

        # restructure
        params = tf.reshape(params, [-1])
        W = tf.reshape(
            params[:self.blocksize],
            shape=[self.n_ensemble, self.input_dim, 1])

        U = tf.reshape(
            params[self.blocksize:2 * self.blocksize],
            shape=[self.n_ensemble, 1, self.input_dim])

        G = tf.sigmoid(
            tf.reshape(
                params[2 * self.blocksize:3 * self.blocksize],
                shape=[self.n_ensemble, 1, self.input_dim]
            )
        )
        U = U * G
        B = tf.reshape(params[3 * self.blocksize:], [self.n_ensemble, 1, 1])
        return [W, B, U]

    def compute_output_shape(self, input_shape):
        W_shape = [self.n_ensemble, self.input_dim, 1]
        B_shape = [self.n_ensemble, 1, 1]
        U_shape = [self.n_ensemble, 1, self.input_dim]
        return W_shape, B_shape, U_shape


class CNF(tf.keras.Model):

    def __init__(self, input_dim, hidden_dim, n_ensemble):
        super().__init__()
        self.hyper_net = HyperNet(input_dim, hidden_dim, n_ensemble)

    def call(self, inputs, **kwargs):
        t, z_p_concat = inputs
        z = z_p_concat[:, :self.hyper_net.input_dim]
        W, B, U = self.hyper_net(t)
        Z = tf.tile(tf.expand_dims(z, 0), [self.hyper_net.n_ensemble, 1, 1])

        with tf.GradientTape() as g:
            g.watch(Z)
            h = tf.tanh(tf.matmul(Z, W) + B)
            dzdt = tf.reduce_mean(tf.matmul(h, U), 0)
            reduced_h = tf.reduce_sum(h)

        dhdZ = g.gradient(
            target=reduced_h,
            sources=Z,
        )
        dlogpz = -tf.matmul(dhdZ, tf.transpose(U, [0, 2, 1]))
        dlogpz = tf.reduce_mean(dlogpz, axis=0)
        return tf.concat([dzdt, dlogpz], axis=1)
