from pprint import pprint
import tensorflow as tf
import datasets
import pickle
import models
import hooks
from tqdm import tqdm

batch_size = 32
lr_schedule = [(1, 0.001), (5, 0.002), (5, 0.001), (5, 0.0001)]
model_dir = '/media/data-disk-slow/jupyter/Research/kkol/deep_learning_notes/subclasses/models/'
model_dir = '../models/test_model'
image_shape = [48, 48, 1]

train_input_func, num_train_examples = datasets.get_input_function(
    "../data/omniglot_training_48_48.pkl",
    batch_size, do_augmentation=True)
valid_input_func, num_valid_examples = datasets.get_input_function(
    "../data/omniglot_validation_48_48.pkl",
    batch_size, do_augmentation=False)

train_epochs = sum([sh[0] for sh in lr_schedule])
steps_per_epoch = num_train_examples // batch_size
steps_per_validation = 32

print(f'train_epochs: {train_epochs}')
print(f'steps_per_epoch: {steps_per_epoch}')
print(f'steps_per_validation: {steps_per_validation}')

class_mapping = pickle.load(open('../data/classes_level0_mapping.pkl', 'rb'))
num_classes = len(class_mapping)

print(f'class_mapping: ')
pprint(class_mapping)
print(f'num_train_examples: {num_train_examples}')
print(f'num_valid_examples: {num_valid_examples}')
print(f'num_classes: {num_classes}')

model_func = models.resnet_model_fn(
    feature_extractor_fn=models.resnet_feature_extractor,
    head_fn=models.classification_head,
    loss_fn=models.classification_loss,
    metrics_fn=models.classification_metrics,
    optimizer_fn=models.adam_optimizer_fn(),
    learning_rate_fn=models.learning_rate_scheduler(
        schedule=lr_schedule,
        batch_size=batch_size,
        num_examples=num_train_examples
    ),
    global_losses_fns=[models.global_averaged_l2_loss(0.001)]
)

session_config = tf.ConfigProto(
    inter_op_parallelism_threads=8,
    intra_op_parallelism_threads=8,
    allow_soft_placement=True)

run_config = tf.estimator.RunConfig().replace(
    save_checkpoints_secs=1e9,
    session_config=session_config)

classifier = tf.estimator.Estimator(
        model_fn=model_func,
        model_dir=model_dir,
        config=run_config,
        params={
            'num_classes': num_classes,
            'image_shape': image_shape
        }
    )
tf.logging.set_verbosity(tf.logging.INFO)

for epoch in tqdm(range(train_epochs)):
    print(f'Starting a training epoch [{epoch}/{train_epochs}]')

    train_hooks = [
        hooks.get_examples_per_second_hook(batch_size=batch_size),
        #hooks.get_profiler_hook(),
        hooks.get_logging_tensor_hook(every_n_iter=10, tensors_to_log=[
            'head_accuracy', 'cross_entropy', 'learning_rate'])
    ]




    classifier.train(
        input_fn=train_input_func,
        hooks=train_hooks,
        steps=steps_per_epoch)


    print('Starting to evaluate')

    # eval_results = classifier.evaluate(
    #     input_fn=train_input_func,
    #     steps=steps_per_validation
    # )
    # print(eval_results)
