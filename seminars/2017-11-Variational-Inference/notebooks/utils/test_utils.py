import tensorlayer as tl
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import edward as ed


def running_mean(x, N):
    return np.convolve(x, np.ones((N,)) / N)[(N - 1):]


def train_mle_network(
        sess, qx, x_input, y_label,
        mnist, batch_size,
        num_epochs=10,
        reshape=[-1, 28 * 28],
        feed_dict={}
):
    optimizer = tf.train.AdamOptimizer(
        learning_rate=0.005, beta1=0.9, beta2=0.999,
    )

    num_iters = int(num_epochs * mnist.train.num_examples // batch_size)

    cost = tl.cost.cross_entropy(qx, y_label, 'cost')
    opt_op = optimizer.minimize(cost)

    correct_prediction = tf.equal(tf.argmax(qx, 1, output_type=tf.int32), y_label)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.global_variables_initializer().run(session=sess)

    hist = {'test': [], 'train': []}
    for it in range(num_iters):
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        x_batch = x_batch.reshape(reshape)

        input_dict = {x_input: x_batch,
                      y_label: y_batch}
        input_dict.update(feed_dict)

        output = sess.run([opt_op, cost, accuracy], feed_dict=input_dict)
        if it % 2000 == 0:
            print(it, num_iters, output)

        hist['train'].append([it] + output)

        if it % 100 == 0:
            x_batch, y_batch = mnist.test.next_batch(batch_size)
            x_batch = x_batch.reshape(reshape)

            input_dict = {x_input: x_batch,
                          y_label: y_batch}
            input_dict.update(feed_dict)

            test_output = sess.run([cost, accuracy], feed_dict=input_dict)

        hist['test'].append([it] + test_output)

    train_hist = np.array(hist['train'])
    test_hist = np.array(hist['test'])

    plt.plot(running_mean(train_hist[:, 2][::4], 50)[:-50], label='train')
    plt.plot(running_mean(test_hist[:, 1][::4], 50)[:-50], label='test')
    plt.yscale('log')
    plt.legend()
    plt.show()
    plt.plot(running_mean(train_hist[:, 3][::4], 50)[:-50], label='train')
    plt.plot(running_mean(test_hist[:, 2][::4], 50)[:-50], label='test')
    plt.legend()
    plt.show()

    return hist


def train_edward_model(
        sess, qx, x_input, y_label, y_pred,
        posteriors,
        kl_scaling,
        label_mapping,
        mnist,
        batch_size,
        num_epochs,
        num_samples=5,
        reshape=[-1, 28 * 28],
        feed_dict={}
):
    # Define the VI inference technique, ie. minimise the KL divergence between q and p.
    inference = ed.KLqp(posteriors, data=label_mapping)
    optimizer = tf.train.AdamOptimizer(
        learning_rate=0.01, beta1=0.9, beta2=0.999,
        epsilon=1e-08, use_locking=False, name="adam_opt"
    )

    num_iters = int(num_epochs * mnist.train.num_examples // batch_size)

    # Initialise the inference variables
    inference.initialize(
        n_iter=num_iters,
        n_samples=num_samples,
        n_print=100,
        scale={y_pred: float(mnist.train.num_examples) / batch_size},
        optimizer=optimizer,
        kl_scaling=kl_scaling
    )

    tf.global_variables_initializer().run()

    cost = tl.cost.cross_entropy(qx, y_label, 'cost')
    correct_prediction = tf.equal(tf.argmax(qx, 1, output_type=tf.int32), y_label)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    hist = []

    for it in range(inference.n_iter):
        x_batch, y_batch = mnist.train.next_batch(batch_size)

        x_batch = x_batch.reshape(reshape)

        input_dict = {x_input: x_batch,
                      y_label: y_batch}

        input_dict.update(feed_dict)
        info_dict = inference.update(feed_dict=input_dict)

        if it % 500 == 0:
            x_batch, y_batch = mnist.test.next_batch(batch_size)
            x_batch = x_batch.reshape(reshape)

            input_dict = {x_input: x_batch,
                          y_label: y_batch}
            input_dict.update(feed_dict)

            test_output = sess.run([cost, accuracy], feed_dict=input_dict)
            hist.append([it] + test_output)
            print(f" test_acc: {test_output[1]}")

        info_dict['cost'] = test_output[0]
        info_dict['accuracy'] = test_output[1]

        inference.print_progress(info_dict)

    return inference, hist


def compute_test_acc(
        sess,
        output_tensor,
        x_input,
        mnist,
        reshape=[-1, 28 * 28],
        feed_dict={}
):
    Y_true = []
    Y_pred = []
    for _ in range(200):
        x_batch, y_batch = mnist.test.next_batch(100)

        x_batch = x_batch.reshape(reshape)

        input_dict = {x_input: x_batch}
        input_dict.update(feed_dict)

        y_batch_pred = sess.run(output_tensor, feed_dict=input_dict)
        Y_true.append(y_batch)
        Y_pred.append(y_batch_pred)

    Y_correct = (np.concatenate(Y_pred).argmax(-1) == np.concatenate(Y_true))
    return sum(Y_correct) / len(Y_correct)


def plot_predictions(
        sess,
        output_tensor,
        x_input,
        mnist,
        only_incorrect=True,
        reshape=[-1, 28 * 28],
        feed_dict={}
):
    X_test = mnist.test.images
    Y_test = mnist.test.labels

    which_picture = np.random.choice(range(Y_test.shape[0]), 1)[0]

    X_sample = np.repeat(np.expand_dims(X_test[which_picture], 0), repeats=32, axis=0)
    Y_sample = np.zeros(shape=[10])
    Y_sample[Y_test[which_picture]] = 1

    prob_output = tf.nn.softmax(output_tensor)

    X_pred = []
    for X in X_sample:
        x_batch = np.expand_dims(X, 0).reshape(reshape)
        input_dict = {x_input: x_batch}
        input_dict.update(feed_dict)
        
        X_pred.append(
            sess.run(prob_output, feed_dict=input_dict)
        )

    X_pred = np.concatenate(X_pred)

    pred_class = X_pred.mean(0).argmax()
    is_correct = pred_class == Y_test[which_picture]

    if not is_correct:
        print("BAD PREDICTION")

    if only_incorrect and is_correct:
        return is_correct

    plt.figure(figsize=(20, 5))
    plt.subplot(121)
    plt.title(f"Correct: {is_correct}")
    plt.imshow(X_sample[0].reshape([28, 28]))

    plt.subplot(122)
    for xi in X_pred:
        plt.plot(xi / xi.max(), color='k', alpha=0.4)

    plt.plot(Y_sample)
    plt.show()

    return is_correct
