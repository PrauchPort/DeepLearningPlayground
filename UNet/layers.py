import tensorflow as tf
from tensorflow.keras import layers


def _crop_and_concat(inputs, residual_input):

    factor = inputs.shape[1] / residual_input.shape[1]
    return tf.concat([inputs, tf.image.central_crop(residual_input, factor)], axis=-1)


class InputBlock(tf.keras.Model):

    def __init__(self, filters):
        super().__init__(self)

        with tf.name_scope('input_block'):
            self.conv1 = layers.Conv2D(filters=filters, kernel_size=(3, 3), activation=tf.nn.relu)

            self.conv2 = layers.Conv2D(filters=filters, kernel_size=(3, 3), activation=tf.nn.relu)

            self.maxpool = layers.MaxPooling2D(pool_size=(2, 2), strides=2)

    def call(self, inputs):
        out = self.conv1(inputs)
        out = self.conv2(out)
        mp = self.maxpool(out)

        return mp, out


class DownsampleBlock(tf.keras.Model):

    def __init__(self, filters, idx):
        super().__init__(self)

        with tf.name_scope('downsample_block_{}'.format(idx)):

            self.conv1 = layers.Conv2D(filters=filters, kernel_size=(3, 3), activation=tf.nn.relu)

            self.conv2 = layers.Conv2D(filters=filters, kernel_size=(3, 3), activation=tf.nn.relu)

            self.maxpool = layers.MaxPooling2D(pool_size=(2, 2), strides=2)

    def call(self, inputs):

        out = self.conv1(inputs)
        out = self.conv2(out)
        mp = self.maxpool(out)

        return mp, out


class BottleneckBlock(tf.keras.Model):

    def __init__(self, filters):
        super().__init__(self)

        with tf.name_scope('bottleneck_block'):

            self.conv1 = layers.Conv2D(filters, kernel_size=(3, 3), activation=tf.nn.relu)

            self.conv2 = layers.Conv2D(filters, kernel_size=(3, 3), activation=tf.nn.relu)

            self.dropout = layers.Dropout(0.5)

            self.conv_t = layers.Conv2DTranspose(
                filters=filters // 2, kernel_size=(3, 3), strides=(2, 2), padding='same',
                activation=tf.nn.relu)

    def call(self, inputs, training):

        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.dropout(out, training=training)
        out = self.conv_t(out)

        return out


class UpsampleBlock(tf.keras.Model):

    def __init__(self, filters, idx):
        super().__init__(self)

        with tf.name_scope('upsample_block_{}'.format(idx)):

            self.conv1 = layers.Conv2D(filters, kernel_size=(3, 3), activation=tf.nn.relu)

            self.conv2 = layers.Conv2D(filters, kernel_size=(3, 3), activation=tf.nn.relu)

            self.conv_t = layers.Conv2DTranspose(
                filters=filters // 2, kernel_size=(3, 3), strides=(2, 2), padding='same',
                activation=tf.nn.relu)

    def call(self, inputs, residual_input):

        out = _crop_and_concat(inputs, residual_input)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv_t(out)

        return out


class OutputBlock(tf.keras.Model):

    def __init__(self, filters, n_classes):
        super().__init__(self)

        with tf.name_scope('output_block'):

            self.conv1 = layers.Conv2D(filters, kernel_size=(3, 3), activation=tf.nn.relu)

            self.conv2 = layers.Conv2D(filters, kernel_size=(3, 3), activation=tf.nn.relu)

            self.conv3 = layers.Conv2D(filters=n_classes, kernel_size=(1, 1), activation=None)

    def call(self, inputs, residual_input):

        out = _crop_and_concat(inputs, residual_input)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        return out
