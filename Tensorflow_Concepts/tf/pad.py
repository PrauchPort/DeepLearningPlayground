import tensorflow as tf
import numpy as np

if __name__ == "__main__":

    t = tf.constant([[1, 2, 3], [4, 5, 6]])
    paddings = tf.constant([[2, 1], [3, 2]])

    print(tf.pad(t, paddings, "CONSTANT").numpy())
