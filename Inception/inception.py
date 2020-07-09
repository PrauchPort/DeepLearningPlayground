import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

import numpy as np


tf.random.set_seed(22)
np.random.seed(22)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

assert tf.__version__.startswith('2.')


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train.astype(np.float32) / 255., x_test.astype(np.float32) / 255.

x_train, x_test = np.expand_dims(x_train, axis=3), np.expand_dims(x_test, axis=3)

db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(256)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(256)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


class ConvBNRelu(keras.Model):

    def __init__(self, no_channels, kernel_size=3, strides=1, padding='same'):
        super(ConvBNRelu, self).__init__()
        self.model = keras.models.Sequential()
        self.model.add(layers.Conv2D(no_channels, kernel_size=kernel_size, strides=strides, padding=padding))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Activation('relu'))

    def call(self, x, training=None):
        x = self.model(x, training=training)

        return x


class InceptionBlock1(keras.Model):
    def __init__(self, no_channels, strides=1):
        super(InceptionBlock1, self).__init__()

        self.no_channels = no_channels
        self.strides = strides

        self.conv1 = ConvBNRelu(no_channels, strides=strides)
        self.conv2 = ConvBNRelu(no_channels, kernel_size=3, strides=strides)
        self.conv3_1 = ConvBNRelu(no_channels, kernel_size=3, strides=strides)
        self.conv3_2 = ConvBNRelu(no_channels, kernel_size=3, strides=strides)

        self.pool = layers.MaxPooling2D(3, strides=1, padding='same')
        self.conv_pool = ConvBNRelu(no_channels, kernel_size=3, strides=strides)

    def call(self, x, training=None):

        self.conv1 = self.conv1(x, training=training)

        self.conv2 = self.conv2(x, training=training)

        self.conv3 = self.conv3_1(x, training=training)
        self.conv3 = self.conv3_2(self.conv3, training=training)

        self.conv4 = self.pool(x)
        self.conv4 = self.conv_pool(self.conv4)

        block = tf.concat([self.conv1, self.conv2, self.conv3, self.conv4], axis=3)

        return block


class Inception(keras.Model):

    def __init__(self, num_layers, init_channels, num_classes, **kwargs):
        super(Inception, self).__init__(**kwargs)

        self.in_channels = init_channels
        self.out_channels = init_channels
        self.init_channels = init_channels
        self.num_classes = num_classes
        self.num_layers = num_layers

        self.conv1 = ConvBNRelu(init_channels, kernel_size=3, strides=1)

        self.blocks = keras.models.Sequential()

        for i in range(self.num_layers):
            for layer_id in range(2):

                if layer_id == 0:
                    conv = ConvBNRelu(self.out_channels, kernel_size=3, strides=2)
                else:
                    conv = ConvBNRelu(self.out_channels, kernel_size=3, strides=1)

                self.blocks.add(conv)
            self.out_channels *= 2

        self.gap = layers.GlobalAveragePooling2D()
        self.dense = layers.Dense(self.num_classes)

    def call(self, x, training=None):
        nn = self.conv1(x, training=training)
        nn = self.blocks(nn, training=training)
        nn = self.gap(nn, training=training)
        nn = self.dense(nn, training=training)

        return nn


def main():
    batch_size = 32

    epochs = 100
    model = Inception(2, 16, 10)

    model.build(input_shape=(None, 28, 28, 1))
    model.summary()

    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    criteon = keras.losses.CategoricalCrossentropy(from_logits=True)

    acc_meter = keras.metrics.Accuracy()

    for epoch in range(10):

        for step, (x, y) in enumerate(db_train):

            with tf.GradientTape() as tape:

                logits = model(x)

                loss = criteon(tf.one_hot(y, depth=10), logits)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 10 == 0:

                print(epoch, step, 'loss: ', loss.numpy())

    acc_meter.reset_states()
    for x, y in db_test:
        logits = model(x, training=False)
        pred = tf.argmax(logits, axis=1)

        acc_meter.update_state(y, pred)

    print(epoch, 'evaluation acc:', acc_meter.result().numpy())


if __name__ == '__main__':
    main()
