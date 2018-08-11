from keras.regularizers import Regularizer
import keras.backend as K


class SeluRegularizer(Regularizer):

    def __init__(self, mu=0.001, tau=0.001):
        self.mu = K.cast_to_floatx(mu)
        self.tau = K.cast_to_floatx(tau)

    def __call__(self, x):
                
        mean_loss = self.mu * K.mean(K.square(K.sum(x, axis=0)))
        tau_loss = - self.tau * K.mean(K.log(K.sum(K.square(x), axis=0) + K.epsilon()))
        
        return mean_loss + tau_loss

    def get_config(self):
        return {'mu': float(self.mu),
                'tau': float(self.tau)}
