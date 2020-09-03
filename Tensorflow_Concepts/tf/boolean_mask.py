import tensorflow as tf
import numpy as np


if __name__ == "__main__":

    tensor = [0, 1, 2, 3]
    mask = np.array([True, False, True, False])

    print(tf.boolean_mask(tensor, mask))

    tensor = [[1, 2], [3, 4], [5, 6]]
    mask = np.array([True, False, True])
    print(tf.boolean_mask(tensor, mask))
