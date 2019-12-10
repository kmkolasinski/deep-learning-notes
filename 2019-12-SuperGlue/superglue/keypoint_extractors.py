"""
Model path taken from: https://github.com/mmmfarrell/SuperPoint/blob/master/pretrained_models/sp_v5.tgz
SuperPoint Implementation: https://github.com/rpautrat/SuperPoint
"""
from typing import List, Tuple

import cv2 as cv
import numpy as np
import tensorflow as tf
from dataclasses import dataclass


def preprocess_image(img: np.ndarray) -> np.ndarray:
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


def extract_superpoint_keypoints_and_descriptors(
    keypoint_map: np.ndarray, descriptor_map: np.ndarray, keep_k_points: int = 1000
) -> Tuple[List[cv.KeyPoint], np.ndarray]:

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
    keypoints = keypoints[:, :2].astype(int)

    # Get descriptors for keypoints
    desc = descriptor_map[keypoints[:, 0], keypoints[:, 1]]

    # Convert from just pts to cv2.KeyPoints
    keypoints = [cv.KeyPoint(p[1], p[0], 100 *d) for p, d in zip(keypoints, keypoints_probs)]
    return keypoints, desc


@dataclass(frozen=True)
class Keypoints:
    keypoints: List[cv.KeyPoint]
    features: np.ndarray

    def numpy_keypoints(self):
        return np.array([(kp.pt[1], kp.pt[0]) for kp in self.keypoints])

    @staticmethod
    def convert_numpy_keypoints_to_cv(points: np.ndarray) -> List[cv.KeyPoint]:
        return [cv.KeyPoint(point[1], point[0], 5) for point in points]


class SuperPointExtractor:

    def __init__(self, saved_model_path: str, k_top_keypoints: int = 1000):
        self.saved_model_path = saved_model_path
        self.imported_model = tf.saved_model.load(saved_model_path, tags=['serve'])
        self.predict_fn = self.imported_model.signatures["serving_default"]
        self.k_top_keypoints = k_top_keypoints

    def extract(self, image: np.ndarray) -> Keypoints:
        img_preprocessed = preprocess_image(image)
        predictions = self.predict_fn(image=tf.constant(img_preprocessed))
        keypoint_map = np.squeeze(predictions["prob_nms"])
        descriptor_map = np.squeeze(predictions["descriptors"])
        cv_keypoints, features = extract_superpoint_keypoints_and_descriptors(
            keypoint_map, descriptor_map, self.k_top_keypoints
        )
        return Keypoints(cv_keypoints, features)
