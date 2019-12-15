import os
import tensorflow as tf
import numpy as np
import superglue.keypoint_extractors as ke

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.enable_eager_execution()


class TestKeypointExtractor(tf.test.TestCase):
    def test_extract_superpoint_keypoints_and_descriptors(self):
        descriptors_map = tf.random.uniform([32, 32, 64])
        keypoints_map = tf.random.uniform([32, 32])

        keypoints, probs, desc = ke.tf_extract_superpoint_keypoints_and_descriptors(keypoints_map, descriptors_map)

        keypoints_np, probs_np, desc_np = ke.py_extract_superpoint_keypoints_and_descriptors(
            keypoints_map.numpy(), descriptors_map.numpy())

        self.assertAllClose(keypoints.numpy(), keypoints_np)
        self.assertAllClose(probs.numpy(), probs_np)
        self.assertAllClose(desc.numpy(), desc_np)

