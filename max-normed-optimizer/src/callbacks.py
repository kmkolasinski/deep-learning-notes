from typing import List, Dict

import keras

_DEFAULT_METRICS = {
    "losses": ["loss", "val_loss"],
    "accuracies": ["acc", "val_acc"]
}


class AggregateMetricsOnBatchEnd(keras.callbacks.Callback):
    def __init__(self, monitor_dict: Dict[str, List[str]] = _DEFAULT_METRICS):
        super(AggregateMetricsOnBatchEnd, self).__init__()
        self.logs = []
        monitor_values = {}
        for key in monitor_dict:
            monitor_values[key] = {}
            for val in monitor_dict[key]:
                monitor_values[key][val] = []
        self.monitor_values = monitor_values
        self.monitor_dict = monitor_dict

    def on_batch_end(self, batch, logs=None):
        self.logs.append(logs)
        for key in self.monitor_dict:
            for val in self.monitor_dict[key]:
                if logs.get(val) is not None:
                    self.monitor_values[key][val].append(logs.get(val))


class AggregateMetricsOnEpochEnd(AggregateMetricsOnBatchEnd):
    def __init__(self, monitor_dict: Dict[str, List[str]] = _DEFAULT_METRICS):
        super(AggregateMetricsOnEpochEnd, self).__init__(monitor_dict)

    def on_batch_end(self, batch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        self.logs.append(logs)
        for key in self.monitor_dict:
            for val in self.monitor_dict[key]:
                if logs.get(val) is not None:
                    self.monitor_values[key][val].append(logs.get(val))