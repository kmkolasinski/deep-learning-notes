import os
import tensorflow as tf
import numpy as np
from scipy.spatial import distance_matrix
import superglue.datasets as datasets

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

find_valid_matches = tf.function(datasets.find_valid_matches)


class TestDatasets(tf.test.TestCase):
    def test_distance_matrix(self):
        melon = tf.random.normal([101, 3])
        musk = tf.random.normal([71, 3])
        dist_matrix = datasets.distance_matrix(melon, musk)
        expected_dist_matrix = distance_matrix(melon.numpy(), musk.numpy())
        self.assertAllClose(dist_matrix.numpy(), expected_dist_matrix, atol=1e-4)

    def test_distance_matrix_no_right_or_left(self):
        melon = tf.random.normal([101, 3])
        musk = tf.random.normal([0, 3])
        dist_matrix = datasets.distance_matrix(melon, musk)
        self.assertEqual(dist_matrix.shape, [101, 0])
        melon = tf.random.normal([0, 3])
        musk = tf.random.normal([70, 3])
        dist_matrix = datasets.distance_matrix(melon, musk)
        self.assertEqual(dist_matrix.shape, [0, 70])

    def test_find_valid_matches(self):

        matrix = np.array(
            [
                [5.0, 1.0, 5.0, 5.0],
                [5.0, 5.0, 5.0, 5.0],
                [1.0, 2.0, 5.0, 5.0],
                [5.0, 5.0, 5.0, 1.0],
                [5.0, 5.0, 5.0, 5.0],
            ]
        ).astype(np.float32)

        ri, ci = datasets.find_valid_matches(
            distance_matrix=matrix, reprojection_threshold=2.5
        )
        self.assertAllEqual(ri, tf.constant([0, 2, 3]))
        self.assertAllEqual(ci, tf.constant([1, 0, 3]))

        ri, ci = datasets.find_valid_matches(
            distance_matrix=matrix, reprojection_threshold=0.0
        )
        self.assertAllEqual(ri, tf.constant([]))
        self.assertAllEqual(ci, tf.constant([]))

    def test_find_valid_matches_no_cols(self):

        ri, ci = datasets.find_valid_matches(tf.zeros([101, 0]))
        self.assertAllEqual(ri, tf.constant([]))
        self.assertAllEqual(ci, tf.constant([]))

        ri, ci = datasets.find_valid_matches(tf.zeros([0, 11]))
        self.assertAllEqual(ri, tf.constant([]))
        self.assertAllEqual(ci, tf.constant([]))

    def test_graph_find_valid_matches(self):

        ri, ci = find_valid_matches(tf.zeros([101, 0]))
        self.assertAllEqual(ri, tf.constant([]))
        self.assertAllEqual(ci, tf.constant([]))

        ri, ci = find_valid_matches(tf.zeros([0, 11]))
        self.assertAllEqual(ri, tf.constant([]))
        self.assertAllEqual(ci, tf.constant([]))

    def test_pad_or_slice(self):

        tensor = tf.random.uniform([11, 4])
        new_tensor = datasets.pad_or_slice(tensor, 3)
        self.assertAllClose(new_tensor[:3], tensor[:3])

        new_tensor = datasets.pad_or_slice(tensor, 13)
        self.assertAllClose(new_tensor[:11], tensor[:11])
        self.assertAllClose(new_tensor[11:], tf.zeros([2, 4]))

        tensor = tf.random.uniform([0, 4])
        new_tensor = datasets.pad_or_slice(tensor, 3)
        self.assertAllClose(new_tensor, tf.zeros([3, 4]))

    def test_create_assignment_matrix(self):
        matrix = datasets.create_assignment_matrix(2, 2)
        exp_matrix = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ])
        self.assertAllEqual(matrix, exp_matrix)

        matrix = datasets.create_assignment_matrix(2, 3)
        exp_matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])
        self.assertAllEqual(matrix, exp_matrix)

        matrix = datasets.create_assignment_matrix(0, 3)
        exp_matrix = np.array([
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [1, 1, 1, 0]
        ])
        self.assertAllEqual(matrix, exp_matrix)

        matrix = datasets.create_assignment_matrix(5, 3)
        exp_matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0]
        ])
        self.assertAllEqual(matrix, exp_matrix)


    def test_estimate_valid_matches(self):
        features_a = tf.random.normal([5, 3])
        keypoints_a = tf.constant([
            [5, 0.0],
            [5, 5],
            [5, 10],
            [5, 20],
            [5, 25],
        ])
        features_b = tf.random.normal([7, 3])
        keypoints_b = tf.constant([
            [5, 25.0],
            [5, 100],
            [5, 5],
            [5, 100],
            [5, 100],
            [5, 100],
            [5, 0],
        ])

        fa, ka, fb, kb, num_matches = datasets.estimate_valid_matches(
            features_a, keypoints_a, features_b, keypoints_b,
            reprojection_threshold = 1
        )
        self.assertEqual(3, num_matches)
        self.assertAllEqual(ka.numpy()[:3], keypoints_a.numpy()[[0, 1, 4], :])
        self.assertAllEqual(kb.numpy()[:3], keypoints_b.numpy()[[6, 2, 0], :])
        self.assertAllEqual(fa.numpy()[:3], features_a.numpy()[[0, 1, 4], :])
        self.assertAllEqual(fb.numpy()[:3], features_b.numpy()[[6, 2, 0], :])

        features, labels = datasets.prepare_training_assignments(
            features_a, keypoints_a, features_b, keypoints_b,
            reprojection_threshold=1, num_matches=10, image_size=(100, 100)
        )
        print(labels)

    def test_estimate_valid_matches_no_matches(self):
        features_a = tf.random.normal([5, 3])
        keypoints_a = tf.constant([
            [5, 0.0],
            [5, 5],
            [5, 10],
            [5, 20],
            [5, 25],
        ])
        features_b = tf.random.normal([7, 3])
        keypoints_b = tf.constant([
            [5, 100.0],
            [5, 100],
            [5, 100],
            [5, 100],
            [5, 100],
            [5, 100],
            [5, 100],
        ])

        fa, ka, fb, kb, num_matches = datasets.estimate_valid_matches(
            features_a, keypoints_a, features_b, keypoints_b,
            reprojection_threshold = 1
        )
        self.assertEqual(0, num_matches)
        self.assertAllEqual(ka.numpy()[:0], tf.zeros([0, 2]))
        self.assertAllEqual(fa.numpy()[:0], tf.zeros([0, 3]))
        self.assertAllEqual(kb.numpy()[:0], tf.zeros([0, 2]))
        self.assertAllEqual(fb.numpy()[:0], tf.zeros([0, 3]))

        features, labels = datasets.prepare_training_assignments(
            features_a, keypoints_a, features_b, keypoints_b,
            reprojection_threshold=1, num_matches=8, image_size=(100, 100)
        )

        print(labels)


