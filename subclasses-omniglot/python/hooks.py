"""Hook that counts examples per second every N steps or seconds."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class ExamplesPerSecondHook(tf.train.SessionRunHook):
    """Hook to print out examples per second.
    Total time is tracked and then divided by the total number of steps
    to get the average step time and then batch_size is used to determine
    the running average of examples per second. The examples per second for the
    most recent interval is also logged.
    """

    def __init__(self,
                 batch_size,
                 every_n_steps=None,
                 every_n_secs=None,
                 warm_steps=0):
        """Initializer for ExamplesPerSecondHook.
        Args:
          batch_size: Total batch size across all workers used to calculate
            examples/second from global time.
          every_n_steps: Log stats every n steps.
          every_n_secs: Log stats every n seconds. Exactly one of the
            `every_n_steps` or `every_n_secs` should be set.
          warm_steps: The number of steps to be skipped before logging and running
            average calculation. warm_steps steps refers to global steps across all
            workers, not on each worker
        Raises:
          ValueError: if neither `every_n_steps` or `every_n_secs` is set, or
          both are set.
        """

        if (every_n_steps is None) == (every_n_secs is None):
            raise ValueError('exactly one of every_n_steps'
                             ' and every_n_secs should be provided.')

        self._timer = tf.train.SecondOrStepTimer(
            every_steps=every_n_steps, every_secs=every_n_secs)

        self._step_train_time = 0
        self._total_steps = 0
        self._batch_size = batch_size
        self._warm_steps = warm_steps

    def begin(self):
        """Called once before using the session to check global step."""
        self._global_step_tensor = tf.train.get_global_step()
        if self._global_step_tensor is None:
            raise RuntimeError(
                'Global step should be created to use StepCounterHook.')

    def before_run(self, run_context):  # pylint: disable=unused-argument
        """Called before each call to run().
        Args:
          run_context: A SessionRunContext object.
        Returns:
          A SessionRunArgs object or None if never triggered.
        """
        return tf.train.SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context,
                  run_values):  # pylint: disable=unused-argument
        """Called after each call to run().
        Args:
          run_context: A SessionRunContext object.
          run_values: A SessionRunValues object.
        """
        global_step = run_values.results

        if self._timer.should_trigger_for_step(
                global_step) and global_step > self._warm_steps:
            elapsed_time, elapsed_steps = self._timer.update_last_triggered_step(
                global_step)
            if elapsed_time is not None:
                self._step_train_time += elapsed_time
                self._total_steps += elapsed_steps

                # average examples per second is based on the total (accumulative)
                # training steps and training time so far
                average_examples_per_sec = self._batch_size * (
                        self._total_steps / self._step_train_time)
                # current examples per second is based on the elapsed training steps
                # and training time per batch
                current_examples_per_sec = self._batch_size * (
                        elapsed_steps / elapsed_time)
                # Current examples/sec followed by average examples/sec
                tf.logging.info(
                    'Batch [%g]:  current exp/sec = %g, average exp/sec = '
                    '%g', self._total_steps, current_examples_per_sec,
                    average_examples_per_sec)



_TENSORS_TO_LOG = dict((x, x) for x in ['learning_rate',
                                        'cross_entropy',
                                        'train_accuracy'])


def get_train_hooks(name_list, **kwargs):
    """Factory for getting a list of TensorFlow hooks for training by name.
    Args:
      name_list: a list of strings to name desired hook classes. Allowed:
        LoggingTensorHook, ProfilerHook, ExamplesPerSecondHook, which are defined
        as keys in HOOKS
      **kwargs: a dictionary of arguments to the hooks.
    Returns:
      list of instantiated hooks, ready to be used in a classifier.train call.
    Raises:
      ValueError: if an unrecognized name is passed.
    """

    if not name_list:
        return []

    train_hooks = []
    for name in name_list:
        hook_name = HOOKS.get(name.strip().lower())
        if hook_name is None:
            raise ValueError(
                'Unrecognized training hook requested: {}'.format(name))
        else:
            train_hooks.append(hook_name(**kwargs))

    return train_hooks


def get_logging_tensor_hook(every_n_iter=100, tensors_to_log=None,
                            **kwargs):  # pylint: disable=unused-argument
    """Function to get LoggingTensorHook.
    Args:
      every_n_iter: `int`, print the values of `tensors` once every N local
        steps taken on the current worker.
      tensors_to_log: List of tensor names or dictionary mapping labels to tensor
        names. If not set, log _TENSORS_TO_LOG by default.
      **kwargs: a dictionary of arguments to LoggingTensorHook.
    Returns:
      Returns a LoggingTensorHook with a standard set of tensors that will be
      printed to stdout.
    """
    if tensors_to_log is None:
        raise ValueError('tensors_to_log is None')

    return tf.train.LoggingTensorHook(
        tensors=tensors_to_log,
        every_n_iter=every_n_iter)


def get_profiler_hook(save_steps=1000,
                      **kwargs):  # pylint: disable=unused-argument
    """Function to get ProfilerHook.
    Args:
      save_steps: `int`, print profile traces every N steps.
      **kwargs: a dictionary of arguments to ProfilerHook.
    Returns:
      Returns a ProfilerHook that writes out timelines that can be loaded into
      profiling tools like chrome://tracing.
    """
    return tf.train.ProfilerHook(save_steps=save_steps)


def get_examples_per_second_hook(every_n_steps=100,
                                 batch_size=128,
                                 warm_steps=5,
                                 **kwargs):  # pylint: disable=unused-argument
    """Function to get ExamplesPerSecondHook.
    Args:
      every_n_steps: `int`, print current and average examples per second every
        N steps.
      batch_size: `int`, total batch size used to calculate examples/second from
        global time.
      warm_steps: skip this number of steps before logging and running average.
      **kwargs: a dictionary of arguments to ExamplesPerSecondHook.
    Returns:
      Returns a ProfilerHook that writes out timelines that can be loaded into
      profiling tools like chrome://tracing.
    """
    return ExamplesPerSecondHook(every_n_steps=every_n_steps,
                                       batch_size=batch_size,
                                       warm_steps=warm_steps)



# A dictionary to map one hook name and its corresponding function
HOOKS = {
    'loggingtensorhook': get_logging_tensor_hook,
    'profilerhook': get_profiler_hook,
    'examplespersecondhook': get_examples_per_second_hook,
}
