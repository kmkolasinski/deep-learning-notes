from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image

from superglue.keypoint_extractors import SuperPointExtractor, Keypoints
from superglue.homographies import (
    homographic_augmentation,
    invert_homography,
    warp_points,
)
import tensorflow_addons as tfa
import tensorflow as tf


def distance_matrix(veca: tf.Tensor, vecb: tf.Tensor):
    """Euclidean distance between to vectors

    Args:
        veca: [N, size]
        vecb: [M, size]

    Returns:
        pairwise distance matrix of size [N, M]
    """
    ra = tf.reduce_sum(tf.square(veca), axis=-1, keepdims=True)
    rb = tf.reduce_sum(tf.square(vecb), axis=-1, keepdims=True)
    rb = tf.transpose(rb)
    ab_dot = tf.matmul(veca, tf.transpose(vecb))
    mat = tf.maximum(ra - 2 * ab_dot + rb, 0.0)
    return tf.sqrt(mat)


def find_valid_matches(
        distance_matrix: tf.Tensor,
        reprojection_threshold: float = 5.0,
        return_unmatched: bool = False
):
    """

    Args:
        distance_matrix: [N, M] matrix
        reprojection_threshold: match distance in pixels

    Returns:
        indices of matched rows
        indices of matches cols
    """
    shape = tf.shape(distance_matrix)
    num_rows, num_cols = shape[0], shape[1]
    if num_rows == 0:
        return tf.constant([], dtype=tf.int32), tf.constant([], dtype=tf.int32)
    if num_cols == 0:
        return tf.constant([], dtype=tf.int32), tf.constant([], dtype=tf.int32)

    row_indices = tf.range(0, num_rows, dtype=tf.int32)

    row_matches = tf.cast(tf.argmin(distance_matrix, axis=-1), tf.int32)
    row_distances = tf.reduce_min(distance_matrix, axis=-1)
    is_valid_match = tf.less(row_distances, reprojection_threshold)

    matched_row_indices = tf.boolean_mask(row_indices, is_valid_match)
    matched_col_indices = tf.boolean_mask(row_matches, is_valid_match)

    if return_unmatched:
        is_invalid_match = tf.greater(row_distances, reprojection_threshold)
        unmatched_row_indices = tf.boolean_mask(row_indices, is_invalid_match)
        unmatched_col_indices = tf.boolean_mask(row_matches, is_invalid_match)
        return matched_row_indices, matched_col_indices, \
               unmatched_row_indices, unmatched_col_indices

    return matched_row_indices, matched_col_indices


def pad_or_slice(tensor: tf.Tensor, target_rows: int) -> tf.Tensor:
    """
    Pad first axis with zeros or take first target_rows
    Args:
        tensor: tensor of shape [num_rows, num_features]
        target_rows:

    Returns:
        tensor [target_rows, num_features]
    """
    num_rows = tf.shape(tensor)[0]
    if target_rows < num_rows:
        return tensor[:target_rows, :]

    num_rows_to_pad = target_rows - num_rows
    return tf.pad(tensor, [[0, num_rows_to_pad], [0, 0]], "CONSTANT")


def create_assignment_matrix(num_valid_matches, num_matches: int):
    num_valid_matches = tf.convert_to_tensor(num_valid_matches)
    num_valid_matches = tf.minimum(num_valid_matches, num_matches)
    zeros = (num_matches) * tf.ones([num_valid_matches + num_matches], dtype=tf.int32)
    row = tf.range(0, num_valid_matches + num_matches)
    col = tf.concat([row[:num_valid_matches], zeros], axis=0)

    make_edge = lambda x, y: tf.transpose(tf.stack([x, y]))
    valid_matches = make_edge(row[:num_valid_matches], row[:num_valid_matches])
    dustbin_matches_a = make_edge(row[num_valid_matches:num_matches], zeros[num_valid_matches:num_matches])
    dustbin_matches_b = make_edge(zeros[num_valid_matches:num_matches], row[num_valid_matches:num_matches])
    indices = tf.concat([valid_matches, dustbin_matches_a, dustbin_matches_b], axis=0)
    assignment_matrix = tf.scatter_nd(indices, tf.ones([tf.shape(indices)[0]]), shape=[num_matches+1, num_matches+1])
    return assignment_matrix


def estimate_valid_matches(
    features_a: tf.Tensor,
    keypoints_a: tf.Tensor,
    features_b: tf.Tensor,
    keypoints_b: tf.Tensor,
    reprojection_keypoints_a: tf.Tensor,
    reprojection_threshold: float = 5.0,
):
    dist_matrix = distance_matrix(keypoints_a, reprojection_keypoints_a)
    indices_a, indices_b, unmatched_indices_a, unmatched_indices_b = \
        find_valid_matches(
            dist_matrix, reprojection_threshold, return_unmatched=True
        )

    uf_a = tf.gather(features_a, unmatched_indices_a)
    uf_b = tf.gather(features_b, unmatched_indices_b)
    uk_a = tf.gather(keypoints_a, unmatched_indices_a)
    uk_b = tf.gather(keypoints_b, unmatched_indices_b)

    features_a = tf.gather(features_a, indices_a)
    features_b = tf.gather(features_b, indices_b)
    keypoints_a = tf.gather(keypoints_a, indices_a)
    keypoints_b = tf.gather(keypoints_b, indices_b)

    features_a = tf.concat([features_a, uf_a], axis=0)
    features_b = tf.concat([features_b, uf_b], axis=0)
    keypoints_a = tf.concat([keypoints_a, uk_a], axis=0)
    keypoints_b = tf.concat([keypoints_b, uk_b], axis=0)

    num_valid_matches = tf.shape(indices_a)[0]
    return features_a, keypoints_a, features_b, keypoints_b, num_valid_matches


def prepare_training_assignments(
    features_a: tf.Tensor,
    keypoints_a: tf.Tensor,
    features_b: tf.Tensor,
    keypoints_b: tf.Tensor,
    reprojection_keypoints_a: tf.Tensor,
    image_size: Tuple[int, int],
    reprojection_threshold: float = 5.0,
    num_matches: int = 512,
):

    features_a, keypoints_a, features_b, keypoints_b, num_valid_matches = \
        estimate_valid_matches(
            features_a, keypoints_a, features_b, keypoints_b,
            reprojection_keypoints_a,
            reprojection_threshold
        )

    num_features = tf.shape(features_a)[0]
    latent_size = tf.shape(features_a)[1]

    image_scale = tf.cast(tf.constant([image_size]), tf.float32)
    features_a = pad_or_slice(features_a, num_matches)
    features_b = pad_or_slice(features_b, num_matches)
    keypoints_a = pad_or_slice(keypoints_a / image_scale, num_matches)
    keypoints_b = pad_or_slice(keypoints_b / image_scale, num_matches)

    random_features_a = tf.random.truncated_normal(shape=[num_matches, latent_size])
    random_features_b = tf.random.uniform(shape=[num_matches, latent_size])
    random_keypoints_a = tf.random.uniform(shape=[num_matches, 2])
    random_keypoints_b = tf.random.uniform(shape=[num_matches, 2])

    mask = pad_or_slice(tf.ones([num_features, 1]), num_matches)

    features_a = features_a * mask + random_features_a * (1.0 - mask)
    features_b = features_b * mask + random_features_b * (1.0 - mask)
    keypoints_a = keypoints_a * mask + random_keypoints_a * (1.0 - mask)
    keypoints_b = keypoints_b * mask + random_keypoints_b * (1.0 - mask)

    features = features_a, keypoints_a, features_b, keypoints_b
    labels = create_assignment_matrix(num_valid_matches, num_matches)
    return features, labels


def create_training_image_pair_sampler_fn(
    extractor: SuperPointExtractor, image_size: Tuple[int, int] = None,
    reprojection_threshold: float = 5, num_matches=512
):
    def sample_fn(image: np.ndarray):
        if image_size is not None:
            image = tf.image.resize(image, size=image_size)

        keypoints = extractor.extract(image)
        fa = keypoints.features
        ka = keypoints.keypoints

        data = {"image": image, "keypoints": keypoints.keypoints}

        transformed_data = homographic_augmentation(
            data, add_homography=True, params={}, valid_border_margin=0
        )
        transformed_image = transformed_data['image']

        transformed_keypoints = extractor.extract(transformed_data["image"])

        fb = transformed_keypoints.features
        kb = transformed_keypoints.keypoints

        inv_homography = invert_homography(
            tf.expand_dims(transformed_data["homography"], 0)
        )
        reprojected_ka = warp_points(
            transformed_keypoints.keypoints, inv_homography[0]
        )
        return transformed_image, prepare_training_assignments(
                features_a=fa,
                keypoints_a=ka,
                features_b=fb,
                keypoints_b= kb,
                reprojection_keypoints_a=reprojected_ka,
                image_size=image_size,
                reprojection_threshold=reprojection_threshold,
                num_matches=num_matches,
        )
    return sample_fn


def superpoint_image_pair_generator(
    images_dir: Path,
    model_dir: Path,
    image_size: Tuple[int, int],
    reprojection_threshold: float = 5,
    num_matches=512
):
    extractor = SuperPointExtractor(model_dir)
    sample_fn = create_training_image_pair_sampler_fn(
        extractor=extractor,
        image_size=image_size,
        reprojection_threshold=reprojection_threshold,
        num_matches=num_matches,
    )
    images = [np.array(Image.open(p).resize(image_size)) for p in list(Path(images_dir).glob("*.jpg"))]

    while True:
        index = np.random.randint(0, len(images))
        image = images[index]
        transformed_image, (features, labels) = sample_fn(image)
        yield features, labels
