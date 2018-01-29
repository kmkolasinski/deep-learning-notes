class SeluDenseRegularizer(Regularizer):
    def __init__(self, mu=0.001):
        self.mu = K.cast_to_floatx(mu)

    def __call__(self, x):
        mean_loss = K.mean(K.square(K.sum(x, axis=0)))
        tau_loss = - K.mean(K.log(K.sum(K.square(x), axis=0) + K.epsilon()))

        return self.mu * (mean_loss + tau_loss)

    def get_config(self):
        return {'mu': float(self.mu)}


class SeluConv2DRegularizer(Regularizer):
    def __init__(self, mu=0.001):
        self.mu = K.cast_to_floatx(mu)

    def __call__(self, x):

        shape = K.int_shape(x)
        num_filters = shape[-1]
        x = K.reshape(x, shape=[-1, num_filters])
        mean_loss = K.mean(K.square(K.sum(x, axis=0)))
        tau_loss = - K.mean(K.log(K.sum(K.square(x), axis=0) + K.epsilon()))

        return self.mu * (mean_loss + tau_loss)

    def get_config(self):
        return {'mu': float(self.mu)}
