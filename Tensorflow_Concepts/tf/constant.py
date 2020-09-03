import tensorflow as tf
import numpy as np

if __name__ == "__main__":

    tf.constant([1, 2, 3, 4])

    a = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)

    print(tf.constant(a))

    v = tf.Variable([0.0])
    with tf.GradientTape() as tape:
        loss = tf.constant(v + v)
    print(tape.gradient(loss, v).numpy())
