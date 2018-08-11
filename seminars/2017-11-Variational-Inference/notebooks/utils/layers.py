import tensorflow as tf
from edward.models import Categorical, Normal, Uniform
import edward as ed
from tensorflow.contrib.slim import add_arg_scope, arg_scope



def bayesian_dense(
        layer,
        qlayer,
        n_units,
        kl_scaling=1.0,
        activation=tf.identity,
        name=None
):
    D = layer.get_shape().as_list()[-1]
    K = n_units
    kernel_shape = [D, K]
    bias_shape = [K]

    # prior
    w = Normal(
        loc=tf.zeros([D, K], name=f'prior/w_dense/loc/{name}'),
        scale=tf.ones([D, K], name=f'prior/w_dense/scale/{name}'))
    b = Normal(
        loc=tf.zeros(K, name=f'prior/b_dense/loc/{name}'),
        scale=tf.ones(K, name=f'prior/b_dense/scale/{name}'))

    # posterior
    def get_kernel(layer_name, initializer=tf.contrib.layers.xavier_initializer()):
        kernel = tf.get_variable(
            shape=kernel_shape,
            name=layer_name,
            initializer=initializer
        )
        return kernel

    def get_bias(layer_name, initializer=tf.contrib.layers.xavier_initializer()):
        kernel = tf.get_variable(
            shape=bias_shape,
            name=layer_name,
            initializer=initializer
        )
        return kernel

    # posterior
    qw = Normal(
        loc=get_kernel(f'qw_dense/loc/{name}'),
        scale=tf.nn.softplus(get_kernel(f'qw_dense/scale/{name}', tf.constant_initializer(-5.0)))
    )

    qb = Normal(
        loc=get_bias(f'qb_dense/loc/{name}', tf.zeros_initializer()),
        scale=tf.nn.softplus(get_bias(f'qb_dense/scale/{name}', tf.constant_initializer(-10.0))))


    # qw = Normal(
    #     loc=tf.Variable(tf.random_normal([D, K]),
    #                     name=f'qw_dense/loc/{name}'),
    #     scale=tf.nn.softplus(tf.Variable(tf.random_normal([D, K]),
    #                                      name=f'qw_dense/scale/{name}')))
    #
    # qb = Normal(
    #     loc=tf.Variable(0.05*tf.random_normal([K]),
    #                     name=f'qb_dense/loc/{name}'),
    #     scale=tf.nn.softplus(tf.Variable(tf.random_normal([K]),
    #                                      name=f'qb_dense/scale/{name}')))

    params = dict(
        input_shape=D,
        n_units=K,
        activation=activation,
        weights=[w, b],
        posteriors={w: qw, b: qb},
        kl_scaling={w: kl_scaling, b: kl_scaling},
        name=name
    )

    y = activation(tf.matmul(layer, w) + b)
    qy = activation(tf.matmul(qlayer, qw) + qb)

    return y, qy, params


def bayesian_categorical(layer, qlayer):
    return Categorical(logits=layer), Categorical(logits=qlayer)


@add_arg_scope
def conv2d(layer, filters, kernel_size, training, activation=tf.nn.relu, dropout=0.0,
           name=None):
    x = tf.layers.conv2d(
        layer,
        filters=filters,
        kernel_size=kernel_size,
        strides=(1, 1),
        padding='same',
        data_format='channels_last',
        dilation_rate=(1, 1),
        activation=activation,
        use_bias=True,
        kernel_initializer=tf.contrib.keras.initializers.glorot_uniform(),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        trainable=True,
        name=name,
        reuse=None
    )
    if dropout > 0:
        x = tf.layers.dropout(x, rate=dropout, training=training)

    return x


@add_arg_scope
def bayesian_conv2d(
        layer,
        qlayer,
        filters,
        kernel_size,
        kl_scaling=1.0,
        activation=tf.nn.relu,
        name=None
):
    # computing shapes
    input_shape = layer.get_shape().as_list()[1:]
    kernel_size = (kernel_size, kernel_size)
    input_dim = input_shape[-1]
    kernel_shape = kernel_size + (input_dim, filters)
    bias_shape = (filters,)

    # prior
    w = Normal(
        loc=tf.zeros(kernel_shape, name=f'prior/w/loc/{name}'),
        scale=tf.ones(kernel_shape, name=f'prior/w/scale/{name}'))
    b = Normal(
        loc=tf.zeros(bias_shape, name=f'prior/b/loc/{name}'),
        scale=tf.ones(bias_shape, name=f'prior/b/scale/{name}'))

    # posterior
    def get_kernel(layer_name, initializer=tf.contrib.layers.xavier_initializer()):
        kernel = tf.get_variable(
            shape=kernel_shape,
            name=layer_name,
            initializer=initializer
        )
        return kernel

    def get_bias(layer_name, initializer=tf.contrib.layers.xavier_initializer()):
        kernel = tf.get_variable(
            shape=bias_shape,
            name=layer_name,
            initializer=initializer
        )
        return kernel

    qw = Normal(
        loc=get_kernel(f'qw/loc/{name}'),
        scale=tf.nn.softplus(get_kernel(f'qw/scale/{name}', tf.constant_initializer(-5.0)))
    )

    qb = Normal(
        loc=get_bias(f'qb/loc/{name}', tf.zeros_initializer()),
        scale=tf.nn.softplus(get_bias(f'qb/scale/{name}', tf.constant_initializer(-10.0))))

    params = dict(
        input_shape=input_shape,
        filters=filters,
        kernel_shape=kernel_shape,
        bias_shape=bias_shape,
        activation=activation,
        weights=[w, b],
        posteriors={w: qw, b: qb},
        named_posteriors={'w': qw, 'b': qb},
        kl_scaling={w: kl_scaling, b: kl_scaling},
        name=name
    )

    def apply_conv(l, kernel, bias):
        y = tf.nn.conv2d(l, kernel, strides=[1, 1, 1, 1], padding='SAME')
        return activation(y + bias)

    y = apply_conv(layer, w, b)
    qy = apply_conv(qlayer, qw, qb)

    return y, qy, params