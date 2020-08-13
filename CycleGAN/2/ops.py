import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.regularizers import l2
import numpy as np
import math
from tensorflow.keras import layers

epsilon = 1e-5


def convolution(filters, kernel_size=3, strides=1, kernel_regularizer=0.0005, padding='same',
                use_bias=False, name='conv'):
    return layers.Conv2D(
        name=name, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
        use_bias=use_bias, kernel_regularizer=l2(kernel_regularizer),
        kernel_initializer=tf.random_normal_initializer(stddev=0.1))


def convolution_t(filters, kernel_size=3, strides=1, kernel_regularizer=0.0005, padding='same',
                  use_bias=False, name='conv'):
    return layers.Conv2DTranspose(
        name=name, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
        use_bias=use_bias, kernel_regularizer=l2(kernel_regularizer),
        kernel_initializer=tf.random_normal_initializer(stddev=0.1))


def bn(name, momentum=0.9):
    return layers.BatchNormalization(name=name, momentum=momentum)


class c7s1_k(keras.Model):
    def __init__(
            self, scope: str = 'c7s1_k', filters: int = 16, weight_decay: float = 0.0005,
            norm: str = 'instance'):
        super(c7s1_k, self).__init__(name=scope)
        self.conv1 = conv(filters=filters, kernel_size=7,
                          kernel_regularizer=weight_decay, padding='valid', name='conv')
