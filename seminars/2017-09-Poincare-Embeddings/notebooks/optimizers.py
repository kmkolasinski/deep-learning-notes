from keras.legacy import interfaces
from keras.optimizers import SGD, Adam
import keras.backend as K


def get_normalization(p):
    p_norm = K.sum(K.square(p), -1, keepdims=True)
    mp = K.square(1 - p_norm)/4.0
    return mp, K.sqrt(p_norm)


def project(p, p_norm):
    p_norm_clip = K.maximum(p_norm, 1.0)
    p_norm_cond = K.cast(p_norm > 1.0, dtype='float') * K.epsilon()    
    return p/p_norm_clip - p_norm_cond


class SGDPoincare(SGD):

    def get_updates(self, params, constraints, loss):
        
        grads = self.get_gradients(loss, params)
        self.updates = []

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))
            self.updates .append(K.update_add(self.iterations, 1))

        # momentum
        shapes = [K.get_variable_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        
        for p, g, m in zip(params, grads, moments):

            normalization, p_norm = get_normalization(p)   
            g = normalization * g
                             
            v = self.momentum * m - lr * g  # velocity
            self.updates.append(K.update(m, v))
            
            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
            else:
                new_p = p + v
                        
            new_p = project(new_p, p_norm)
                        
            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates
        
  
class AdamPoincare(Adam):

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))

        t = self.iterations + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        shapes = [K.get_variable_shape(p) for p in params]
        ms = [K.zeros(shape) for shape in shapes]
        vs = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):
            
            normalization, p_norm = get_normalization(p)
            g = normalization * g
            
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
                  
            new_p = project(p_t, p_norm)
            
            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)
                
            self.updates.append(K.update(p, new_p))
        return self.updates
