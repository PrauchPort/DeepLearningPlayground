import tensorflow as tf


if __name__ == "__main__":

    t = tf.constant([[-10., -1., 0.], [0., 2., 10.]])

    t2 = tf.clip_by_value(t, clip_value_min=-1, clip_value_max=1)

    print(t2.numpy())
