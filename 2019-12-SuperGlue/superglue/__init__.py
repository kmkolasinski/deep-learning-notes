import tensorflow as tf
TF_MAJOR_VERSION = tf.__version__.split(".")[0]

IS_TF_2_0 = TF_MAJOR_VERSION == "2"
