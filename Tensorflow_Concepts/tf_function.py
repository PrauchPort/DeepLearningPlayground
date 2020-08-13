import tensorflow as tf
import numpy as np


@tf.function
def f1(x, y):
    return x ** 2 + y


@tf.function
def f2(x):
    if tf.reduce_sum(x) > 0:
        return x * x
    else:
        return -x // 2


@tf.function
def f3():
    return x ** 2 + y


@tf.function
def f4(x):
    for i in tf.range(x):
        v.assign_add(i)


@tf.function
def f5(x):
    for i in x:
        l.append(i + 1)


@tf.function
def f6(x):
    ta = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    for i in range(len(x)):
        ta = ta.write(i, x[i] + 1)
    return ta.stack()


@tf.function
def f7(x):
    return tf.abs(x)


@tf.function
def f8(x):
    return x + 1


if __name__ == "__main__":

    x = tf.constant([2, 3])
    y = tf.constant([3, -2])

    print(f1(np.array([2, 3]), np.array([3, -2])).numpy())

    print(f2(tf.constant(-2)))

    x = tf.constant([-2, -3])
    y = tf.Variable([3, -2])

    print(f3())

    v = tf.Variable(1)

    print(f4(4))
    print(v)

    l = []

    f5(tf.constant([1, 2, 3]))

    print(l)

    print(f6(tf.constant([1, 2, 3])))

    f1 = f7.get_concrete_function(1)
    f2 = f7.get_concrete_function(2)

    print(f1 is f2)

    f1 = f7.get_concrete_function(tf.constant(1))
    f2 = f7.get_concrete_function(tf.constant(2))

    print(f1 is f2)

    vector = tf.constant([1.0, 1.0])
    matrix = tf.constant([3.0])

    print(f8.get_concrete_function(vector) is f8.get_concrete_function(matrix))
