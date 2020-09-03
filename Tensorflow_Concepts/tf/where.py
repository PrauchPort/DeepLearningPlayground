import tensorflow as tf

if __name__ == "__main__":

    a = tf.where([True, False, False, True], [1, 2, 3, 4], [100, 200, 300, 400])
    print(a.numpy())

    b = tf.where([True, False, False, True], [1, 2, 3, 4], [100])
    print(b.numpy())

    c = tf.where([True, False, False, True], [1, 2, 3, 4], 100)
    print(c.numpy())

    d = tf.where([True, False, False, True], 1, 100)
    print(d.numpy())
