import tensorflow as tf

if __name__ == "__main__":

    a = tf.constant([1, 10, 26.9, 2.8, 166.32, 62.3], dtype=tf.float32)

    b = tf.argsort(a, axis=-1, direction='ASCENDING', stable=False, name=None)

    c = tf.keras.backend.eval(b)

    print(b)
    print(c)
