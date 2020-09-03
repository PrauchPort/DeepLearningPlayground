import tensorflow as tf
import numpy as np

x = tf.Variable(1.0, name='x')
z = tf.Variable(3.0, name='z')

with tf.GradientTape() as tape:
    # y will evaluate to 9.0
    y = tf.grad_pass_through(x.assign)(z**2)

grads = tape.gradient(y, z)

print(y.numpy())
print(grads.numpy())
