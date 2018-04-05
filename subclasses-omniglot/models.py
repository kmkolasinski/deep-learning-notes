from collections import namedtuple

import numpy as np
import tensorflow as tf

X_FEATURE = 'image'
Y_LABEL = 'label'


def resnet_feature_extractor(features, mode, params=None):
    """Builds a residual network."""

    # Configurations for each bottleneck group.
    BottleneckGroup = namedtuple('BottleneckGroup',
                                 ['num_blocks', 'num_filters',
                                  'bottleneck_size'])
    groups = [
        BottleneckGroup(3, 32, 32),
        BottleneckGroup(3, 64, 64),
        BottleneckGroup(3, 128, 128)
    ]

    image_shape = params['image_shape']
    x = features[X_FEATURE]
    x = tf.reshape(x, [-1, image_shape[0], image_shape[1], image_shape[2]])
    print('input_image:', x)

    training = (mode == tf.estimator.ModeKeys.TRAIN)

    # First convolution expands to 64 channels
    with tf.variable_scope('conv_layer1'):
        net = tf.layers.conv2d(
            x,
            filters=64,
            kernel_size=5,
            activation=tf.nn.relu)
        net = tf.layers.batch_normalization(net, training=training)

    # Max pool
    net = tf.layers.max_pooling2d(
        net, pool_size=3, strides=2, padding='same')

    # First chain of resnets
    with tf.variable_scope('conv_layer2'):
        net = tf.layers.conv2d(
            net,
            filters=groups[0].num_filters,
            kernel_size=1,
            padding='valid')

    # Create the bottleneck groups, each of which contains `num_blocks`
    # bottleneck groups.
    for group_i, group in enumerate(groups):
        print(f'group {group_i} with shape: {net.shape}')
        net = tf.layers.max_pooling2d(
            net, pool_size=2, strides=2, padding='same')

        for block_i in range(group.num_blocks):
            name = 'group_%d/block_%d' % (group_i, block_i)

            # 1x1 convolution responsible for reducing dimension
            with tf.variable_scope(name + '/conv_in'):
                conv = tf.layers.conv2d(
                    net,
                    filters=group.num_filters,
                    kernel_size=1,
                    padding='valid',
                    activation=tf.nn.relu)
                conv = tf.layers.batch_normalization(conv, training=training)

            with tf.variable_scope(name + '/conv_bottleneck'):
                conv = tf.layers.conv2d(
                    conv,
                    filters=group.bottleneck_size,
                    kernel_size=3,
                    padding='same',
                    activation=tf.nn.relu)
                conv = tf.layers.batch_normalization(conv, training=training)

            # 1x1 convolution responsible for restoring dimension
            with tf.variable_scope(name + '/conv_out'):
                input_dim = net.get_shape()[-1].value
                conv = tf.layers.conv2d(
                    conv,
                    filters=input_dim,
                    kernel_size=1,
                    padding='valid',
                    activation=tf.nn.relu)
                conv = tf.layers.batch_normalization(conv, training=training)

            # shortcut connections that turn the network into its counterpart
            # residual function (identity shortcut)
            net = conv + net

        try:
            # upscale to the next group size
            next_group = groups[group_i + 1]
            with tf.variable_scope('block_%d/conv_upscale' % group_i):
                net = tf.layers.conv2d(
                    net,
                    filters=next_group.num_filters,
                    kernel_size=1,
                    padding='same',
                    activation=None,
                    bias_initializer=None)
        except IndexError:
            pass

    print("Trainable params:", count_number_trainable_params())
    feature_map_output = {'feature_map': net}
    return feature_map_output


def count_number_trainable_params():
    '''
    Counts the number of trainable variables.
    '''
    tot_nb_params = 0
    for trainable_variable in tf.trainable_variables():
        shape = trainable_variable.get_shape()
        current_nb_params = int(np.prod(shape))
        tot_nb_params = tot_nb_params + current_nb_params
    return tot_nb_params


def classification_head(feature_map_output, mode=None, params=None):
    feature_map = feature_map_output['feature_map']

    net_shape = feature_map.get_shape().as_list()
    net = tf.nn.avg_pool(
        feature_map,
        ksize=[1, net_shape[1], net_shape[2], 1],
        strides=[1, 1, 1, 1],
        padding='VALID')

    net_shape = net.get_shape().as_list()
    net = tf.reshape(net, [-1, net_shape[1] * net_shape[2] * net_shape[3]])

    # Compute logits (1 per class) and compute loss.
    num_classes = params['num_classes']
    logits = tf.layers.dense(
        net, num_classes, activation=None, name='head_logits')

    head_output = {
        'logits': logits,
        'classes': tf.argmax(logits, axis=1, name='head_classes'),
        'probabilities': tf.nn.softmax(logits, name='head_probabilities')
    }

    return head_output


def classification_loss(head_output, labels, mode=None, params=None):
    logits = head_output['logits']

    # Calculate loss, which includes softmax cross entropy and L2 regularization
    cross_entropy = tf.losses.softmax_cross_entropy(
        logits=logits, onehot_labels=labels[Y_LABEL])

    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    return cross_entropy


def global_averaged_l2_loss(weight_decay, loss_filter_fn=None):
    def global_func():
        # If no loss_filter_fn is passed, assume we want the default behavior,
        # which is that batch_normalization variables are excluded from loss.
        def exclude_batch_norm(name):
            return 'batch_normalization' not in name

        filter_fn = loss_filter_fn or exclude_batch_norm

        # Add weight decay to the loss.
        l2_loss = weight_decay * tf.add_n(
            [tf.reduce_mean(tf.square(v)) for v in tf.trainable_variables()
             if filter_fn(v.name)])

        tf.summary.scalar('averaged_l2_loss', l2_loss)
        return l2_loss

    return global_func


def classification_metrics(head_output, labels, mode=None, params=None):
    classes = head_output['classes']

    accuracy = tf.metrics.accuracy(
        tf.argmax(labels[Y_LABEL], axis=1), classes)

    metrics = {'accuracy': accuracy}
    # Create a tensor named train_accuracy for logging purposes
    tf.identity(accuracy[1], name='head_accuracy')
    tf.summary.scalar('head_accuracy', accuracy[1])

    return metrics


def adam_optimizer_fn(beta1=0.8, beta2=0.999):
    def func(loss, learning_rate_fn):
        global_step = tf.train.get_or_create_global_step()
        learning_rate = learning_rate_fn(global_step)

        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate,
            beta1=beta1, beta2=beta2)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(optimizer.minimize(loss, global_step), update_ops)
        return train_op

    return func


def learning_rate_scheduler(
        schedule=[(1, 0.01), (2, 0.001)],
        batch_size=32, num_examples=1000
):
    batches_per_epoch = num_examples / batch_size

    epoch_lr = []
    for sh in schedule:
        for i in range(sh[0]):
            epoch_lr.append(sh[1])

    boundaries = []
    vals = [epoch_lr[0]]
    for epoch, lr in enumerate(epoch_lr):
        if epoch == 0:
            continue
        step = batches_per_epoch * epoch
        boundaries.append(int(step))
        vals.append(lr)

    def learning_rate_fn(global_step):
        global_step = tf.cast(global_step, tf.int32)
        return tf.train.piecewise_constant(global_step, boundaries, vals)

    return learning_rate_fn


def resnet_model_fn(
        feature_extractor_fn,
        head_fn,
        loss_fn,
        metrics_fn,
        optimizer_fn,
        learning_rate_fn,
        global_losses_fns=None
):
    def input_func(features,
                   labels,
                   mode,
                   params):
        fm_outputs = feature_extractor_fn(features, mode=mode, params=params)
        head_outputs = head_fn(fm_outputs, mode=mode, params=params)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=head_outputs,
                export_outputs={
                    'predict': tf.estimator.export.PredictOutput(head_outputs)
                })

        head_loss = loss_fn(head_outputs, labels, mode=mode, params=params)
        loss = head_loss
        if global_losses_fns is not None:
            for global_loss_fn in global_losses_fns:
                loss = loss + global_loss_fn()

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimizer_fn(loss, learning_rate_fn)
        else:
            train_op = None

        metrics = metrics_fn(head_outputs, labels, mode=mode, params=params)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=head_outputs,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics)

    return input_func
