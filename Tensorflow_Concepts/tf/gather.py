import tensorflow as tf
import numpy as np

if __name__ == "__main__":

    a = tf.constant(np.array([1, 2, 3, 4, 5, 6]), tf.float32)

    gathered = tf.gather(a, [0, 3, 0, 0])

    print(gathered.numpy())
