import tensorflow as tf

if __name__ == "__main__":

    a = tf.fill([2, 5], 11)

    print(a.numpy())
