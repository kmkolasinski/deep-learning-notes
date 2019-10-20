from typing import Dict
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

keras = tf.keras


def plt_imshow(image: np.ndarray):
    if image.shape[-1] == 3:
        return plt.imshow(image)
    else:
        return plt.imshow(image[..., 0])


def plot_image_pairs(
    features: Dict[str, tf.Tensor], labels: Dict[str, tf.Tensor], num_examples: int = 5
) -> None:
    for i in range(num_examples):
        plt.subplot(121)
        plt.title(labels["label"].numpy()[i])
        plt_imshow(features["image"][i, ...])        
        if "tf_image" in features:
            plt.subplot(122)
            plt_imshow(features["tf_image"][i, ...])
        plt.show()


def plot_probabilities_grid(
    iic_model: tf.keras.Model, dataset_iterator: tf.data.Dataset, num_steps: int = 10
):
    prediction_dict = defaultdict(list)

    for i in range(num_steps):
        features, labels = next(dataset_iterator)
        p_out_preds = iic_model.predict(features, steps=None)
        if type(p_out_preds) == np.ndarray:
            p_out_preds = [p_out_preds]

        for k, p_out in enumerate(p_out_preds):
            prediction_dict[f"y_pred_{k}"] += p_out.argmax(-1).tolist()
        prediction_dict[f"y_true"] += labels["label"].numpy().tolist()

    df = pd.DataFrame(prediction_dict)
    g = sns.PairGrid(df)
    g.map_diag(sns.kdeplot)
    return g.map_offdiag(sns.kdeplot, n_levels=6)


def unsupervised_labels(y, y_hat, num_classes, num_clusters):
    """
    :param y: true label
    :param y_hat: concentration parameter
    :param num_classes: number of classes (determined by data)
    :param num_clusters: number of clusters (determined by model)
    :return: classification error rate
    """
    assert num_classes == num_clusters

    # initialize count matrix
    cnt_mtx = np.zeros([num_classes, num_classes])

    # fill in matrix
    for i in range(len(y)):
        cnt_mtx[int(y_hat[i]), int(y[i])] += 1

    # find optimal permutation
    row_ind, col_ind = linear_sum_assignment(-cnt_mtx)

    # compute error
    error = 1 - cnt_mtx[row_ind, col_ind].sum() / cnt_mtx.sum()
    return error


class PredictionsHistory(keras.callbacks.Callback):
    def __init__(self, validation_data=(), interval: int = 20):
        super().__init__()
        self.interval = interval
        self.data = validation_data
        self.heads_y_pred = defaultdict(list)
        self.y_true = []

    def on_train_begin(self, logs=None):
        self.y_pred = []
        self.y_true = []

    def on_batch_end(self, batch, logs=None):
        if batch % self.interval == 0:
            features, labels = next(self.data)
            y_true = labels["label"].numpy()
            heads_p_pred = self.model.predict(features)
            if type(heads_p_pred) == np.ndarray:
                heads_p_pred = [heads_p_pred]

            for k, p_pred in enumerate(heads_p_pred):
                y_pred = p_pred.argmax(-1)
                self.heads_y_pred[k] += y_pred.tolist()

            self.y_true += y_true.tolist()

    def on_epoch_end(self, epoch, logs=None):
        error_string = " "
        count = len(self.y_true)
        for k, y_pred in self.heads_y_pred.items():
            error = unsupervised_labels(self.y_true, y_pred, 10, 10)
            error_string += f" head[{k}]: {error:.4f} [{count}]"
        print(error_string)

        self.heads_y_pred = defaultdict(list)
        self.y_true = []
