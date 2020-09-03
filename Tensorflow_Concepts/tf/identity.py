import tensorflow as tf
import numpy as np

if __name__ == "__main__":

    a = tf.Variable(0.78)
    a_identity = tf.identity(a)

    a.assign_add(1)

    print(a.numpy())
    print(a_identity.numpy())
