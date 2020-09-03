import tensorflow as tf

if __name__ == "__main__":

    x = tf.constant([1, 2, 3])
    y = tf.broadcast_to(x, [4, 3])

    print(y)
