# inspired by loss of VAEs
def fc_selu_reg(x, mu):
    # average over filter size
    mean = K.mean(x, axis=0)
    tau_sqr = K.mean(K.square(x), axis=0)
    # average over batch size
    mean_loss = K.mean(K.square(mean))
    tau_loss = K.mean(tau_sqr - K.log(tau_sqr + K.epsilon()))
    return mu * (mean_loss + tau_loss)    


class SeluDenseRegularizer(Regularizer):
    def __init__(self, mu=0.001):
        self.mu = K.cast_to_floatx(mu)

    def __call__(self, x):
        return fc_selu_reg(x, self.mu) 

    def get_config(self):
        return {'mu': float(self.mu)}


class SeluConv2DRegularizer(Regularizer):
    def __init__(self, mu=0.001):
        self.mu = K.cast_to_floatx(mu)

    def __call__(self, x):

        shape = K.int_shape(x)
        num_filters = shape[-1]
        x = K.reshape(x, shape=[-1, num_filters])
        return fc_selu_reg(x, self.mu) 

    def get_config(self):
        return {'mu': float(self.mu)}
