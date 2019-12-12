"""
Model path taken from: https://github.com/mmmfarrell/SuperPoint/blob/master/pretrained_models/sp_v5.tgz
SuperPoint Implementation: https://github.com/rpautrat/SuperPoint
"""
from typing import List, Tuple, Union

import cv2 as cv
import numpy as np
import tensorflow as tf
from dataclasses import dataclass


def py_preprocess_image(img: np.ndarray) -> np.ndarray:
    """Prepare input image to form accepted SuperPoint predictor

    Args:
        img: an uint8 image of shape [height, width, 3]

    Returns:
        a processed gray scale image of shape [1, height, width, 1] and dtype=float32
    """
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = np.expand_dims(img, 2)
    img = img.astype(np.float32)
    img_preprocessed = np.expand_dims(img, 0)
    return img_preprocessed


def tf_preprocess_image(img: tf.Tensor) -> tf.Tensor:
    """Prepare input image to form accepted SuperPoint predictor

    Args:
        img: an uint8 image tf.Tensor of shape [height, width, 3]

    Returns:
        a processed gray scale image of shape [1, height, width, 1] and dtype=tf.float32
    """
    img = tf.reduce_mean(tf.cast(img, tf.float32), axis=-1, keepdims=True)
    img_preprocessed = tf.expand_dims(img, 0)
    return img_preprocessed


def preprocess_image(image):
    if type(image) == np.ndarray:
        return py_preprocess_image(image)
    return tf_preprocess_image(image)


def extract_superpoint_keypoints_and_descriptors(
    keypoint_map: np.ndarray, descriptor_map: np.ndarray, keep_k_points: int = 1000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    def select_k_best(points, k):
        """ Select the k most probable points (and strip their proba).
        points has shape (num_points, 3) where the last coordinate is the proba. """
        sorted_prob = points[points[:, 2].argsort(), :3]
        start = min(k, points.shape[0])
        return sorted_prob[-start:, :][::-1, :]

    # Extract keypoints
    keypoints = np.where(keypoint_map >= 1e-6)
    prob = keypoint_map[keypoints[0], keypoints[1]]
    keypoints = np.stack([keypoints[0], keypoints[1], prob], axis=-1)
    keypoints = select_k_best(keypoints, keep_k_points)
    keypoints_probs = keypoints[:, 2]
    int_keypoints = keypoints[:, :2].astype(int)

    # Get descriptors for keypoints
    desc = descriptor_map[int_keypoints[:, 0], int_keypoints[:, 1]]

    # Convert from just pts to cv2.KeyPoints
    keypoints = keypoints[:, :2].astype(np.float32)
    keypoints_probs = keypoints_probs.astype(np.float32)
    return keypoints, keypoints_probs, desc.astype(np.float32)


@dataclass(frozen=True)
class Keypoints:
    keypoints: np.ndarray
    probs: np.ndarray
    features: np.ndarray

    @property
    def cv_keypoints(self) -> List[cv.KeyPoint]:
        return [
            cv.KeyPoint(p[1], p[0], 100 * d) for p, d in zip(self.keypoints, self.probs)
        ]

    @staticmethod
    def convert_numpy_keypoints_to_cv(points: np.ndarray) -> List[cv.KeyPoint]:
        return [cv.KeyPoint(point[1], point[0], 5) for point in points]


class SuperPointExtractor:
    def __init__(self, saved_model_path: str, k_top_keypoints: int = 1000):
        self.saved_model_path = saved_model_path
        self.imported_model = tf.saved_model.load(saved_model_path, tags=["serve"])
        self.predict_fn = self.imported_model.signatures["serving_default"]
        self.k_top_keypoints = k_top_keypoints

    def extract(self, image: Union[np.ndarray, tf.Tensor]) -> Keypoints:
        img_preprocessed = preprocess_image(image)
        predictions = self.predict_fn(image=tf.constant(img_preprocessed))
        keypoint_map = np.squeeze(predictions["prob_nms"])
        descriptor_map = np.squeeze(predictions["descriptors"])
        keypoints, probs, features = extract_superpoint_keypoints_and_descriptors(
            keypoint_map, descriptor_map, self.k_top_keypoints
        )
        return Keypoints(keypoints, probs, features)
