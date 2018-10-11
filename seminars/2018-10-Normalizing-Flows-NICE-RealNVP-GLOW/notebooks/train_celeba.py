"""
WARNING:
This script is supposed to train normalizing flow on Celeba dataset
however, it seems, that it contains some bug since I could not
obtain the same results as in the notebooks. The definition of the
estimator is strictly copied from notebooks.

NOTE:
     * once the bug is solved one can used experiments/train_celeba_TEMPLATE.sh
       script to train models without notebooks.
"""

import argparse
from pprint import pprint

import numpy as np
import tensorflow as tf
from tensorflow.python.training.session_run_hook import SessionRunHook
from tqdm import tqdm

import flow_layers as fl
import nets
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--image_size', default=24, type=int, help='image_size')
parser.add_argument('--train_steps', default=100000, type=int,
                    help='number of training steps')
parser.add_argument('--eval_steps', default=100, type=int,
                    help='number of eval steps')
parser.add_argument('--dataset_path', type=str,
                    default="./datasets/celeba/celeba_valid.tfrecords",
                    help='path to tfrecords')
parser.add_argument('--model_dir', type=str, default="./model",
                    help='save path')
parser.add_argument('--mode', type=str, help='[train|eval]')
parser.add_argument('--l2_reg', type=float, default=0.0001, help='l2 reg')

# sampling - evaluation
parser.add_argument('--sample_beta', type=float, default=0.9,
                    help='sample_beta')
parser.add_argument('--save_secs', type=int, default=100, help='save_secs')
# scheduler
parser.add_argument('--lr', type=float, default=0.001, help='initial lr')
parser.add_argument('--clip', type=float, default=0.0, help='clip gradients')
parser.add_argument('--decay_steps', type=int, default=10000,
                    help='decay_steps')
parser.add_argument('--decay_rate', type=float, default=0.5, help='decay_rate')
# coupling layer NN
parser.add_argument('--width', type=int, default=128)

# flow
parser.add_argument('--num_steps', type=int, default=24)
parser.add_argument('--num_scales', type=int, default=3)
parser.add_argument('--num_bits', type=int, default=5)


ACTNORM_INIT_OPS = "ACTNORM_INIT_OPS"


class InitActnorms(SessionRunHook):

    def after_create_session(self, session, coord):
        # When this is called, the graph is finalized and
        # ops can no longer be added to the graph.
        init_ops = tf.get_collection(ACTNORM_INIT_OPS)
        print("Initializing actnorms")
        num_steps = 30

        for init_op in tqdm(init_ops):
            for i in range(num_steps):
                session.run(init_op)


def main(argv):
    args = parser.parse_args(argv[1:])
    pprint(args)
    batch_size = args.batch_size
    l2_reg = args.l2_reg
    image_size = args.image_size
    sample_beta = args.sample_beta

    def get_train_input_fn():
        x_train_samples = utils.create_tfrecord_dataset_iterator(
            args.dataset_path, batch_size=batch_size,
            image_size=args.image_size
        )
        return {"images": x_train_samples}, {}

    def get_eval_input_fn():
        x_valid_samples = utils.create_tfrecord_dataset_iterator(
            args.dataset_path, batch_size=batch_size, buffer_size=0,
            image_size=args.image_size
        )
        return {"images": x_valid_samples}, {}

    def model_fn(features, labels, mode, params):

        nn_template_fn = nets.OpenAITemplate(
            width=args.width
        )

        layers, actnorm_layers = nets.create_simple_flow(
            num_steps=args.num_steps,
            num_scales=args.num_scales,
            num_bits=args.num_bits,
            template_fn=nn_template_fn
        )

        images = features["images"]
        flow = fl.InputLayer(images)
        model_flow = fl.ChainLayer(layers)
        output_flow = model_flow(flow, forward=True)
        y, logdet, z = output_flow

        for layer in actnorm_layers:
            init_op = layer.get_ddi_init_ops(30)
            tf.add_to_collection(ACTNORM_INIT_OPS, init_op)

        total_params = 0
        trainable_variables = tf.trainable_variables()
        for k, v in enumerate(trainable_variables):
            num_params = np.prod(v.shape.as_list())
            total_params += num_params

        print(f"TOTAL PARAMS: {total_params/1e6} [M]")

        if mode == tf.estimator.ModeKeys.PREDICT:
            raise NotImplementedError()

        tfd = tf.contrib.distributions
        y_flatten = tf.reshape(y, [batch_size, -1])
        z_flatten = tf.reshape(z, [batch_size, -1])

        prior_y = tfd.MultivariateNormalDiag(loc=tf.zeros_like(y_flatten),
                                             scale_diag=tf.ones_like(y_flatten))
        prior_z = tfd.MultivariateNormalDiag(loc=tf.zeros_like(z_flatten),
                                             scale_diag=tf.ones_like(z_flatten))

        log_prob_y = prior_y.log_prob(y_flatten)
        log_prob_z = prior_z.log_prob(z_flatten)

        loss = log_prob_y + log_prob_z + logdet
        # compute loss per pixel, the final loss should be same
        # for different input sizes
        loss = - tf.reduce_mean(loss) / image_size / image_size

        trainable_variables = tf.trainable_variables()
        l2_loss = l2_reg * tf.add_n(
            [tf.nn.l2_loss(v) for v in trainable_variables])

        total_loss = l2_loss + loss

        tf.summary.scalar('total_loss', total_loss)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('l2_loss', l2_loss)

        # Sampling during training and evaluation
        prior_y = tfd.MultivariateNormalDiag(loc=tf.zeros_like(y_flatten),
                                             scale_diag=sample_beta * tf.ones_like(y_flatten))
        prior_z = tfd.MultivariateNormalDiag(loc=tf.zeros_like(z_flatten),
                                             scale_diag=sample_beta * tf.ones_like(z_flatten))

        sample_y_flatten = prior_y.sample()
        sample_y = tf.reshape(sample_y_flatten, y.shape.as_list())
        sample_z = tf.reshape(prior_z.sample(), z.shape.as_list())
        sampled_logdet = prior_y.log_prob(sample_y_flatten)

        inverse_flow = sample_y, sampled_logdet, sample_z
        sampled_flow = model_flow(inverse_flow, forward=False)
        x_flow_sampled, _, _ = sampled_flow
        # convert to uint8
        quantize_image_layer = layers[0]
        x_flow_sampled_uint = quantize_image_layer.to_uint8(x_flow_sampled)

        grid_image = tf.contrib.gan.eval.image_grid(
            x_flow_sampled_uint,
            grid_shape=[4, batch_size // 4],
            image_shape=(image_size, image_size),
            num_channels=3
        )

        grid_summary = tf.summary.image(
            f'samples{sample_beta}', grid_image, max_outputs=10
        )

        if mode == tf.estimator.ModeKeys.EVAL:
            eval_summary_hook = tf.train.SummarySaverHook(
                save_steps=1,
                output_dir=args.model_dir + "/eval",
                summary_op=grid_summary
            )

            return tf.estimator.EstimatorSpec(
                mode,
                loss=total_loss,
                evaluation_hooks=[eval_summary_hook]
            )

        # Create training op.
        assert mode == tf.estimator.ModeKeys.TRAIN

        train_summary_hook = tf.train.SummarySaverHook(
            save_secs=args.save_secs,
            output_dir=args.model_dir,
            summary_op=grid_summary
        )

        global_step = tf.train.get_global_step()
        learning_rate = tf.train.inverse_time_decay(
            args.lr, global_step, args.decay_steps, args.decay_rate,
            staircase=True
        )

        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
        if args.clip > 0.0:
            gvs = optimizer.compute_gradients(total_loss)
            capped_gvs = [
                (tf.clip_by_value(grad, -args.clip, args.clip), var) for grad, var in gvs
            ]
            train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)
        else:
            train_op = optimizer.minimize(total_loss, global_step=global_step)

        return tf.estimator.EstimatorSpec(
            mode, loss=total_loss,
            train_op=train_op, training_hooks=[train_summary_hook]
        )

    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=args.model_dir,
    )

    if args.mode == "train":
        classifier.train(
            input_fn=get_train_input_fn,
            steps=args.train_steps,
            hooks=[InitActnorms()]
        )
        classifier.evaluate(
            input_fn=get_eval_input_fn,
            steps=args.eval_steps
        )

    else:
        eval_result = classifier.evaluate(
            input_fn=get_eval_input_fn,
            steps=args.eval_steps
        )


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
