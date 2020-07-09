import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, GlobalAveragePooling2D, Add
import numpy as np


assert tf.__version__.startswith('2.'), "Tensorflow 2 is required."

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

x_train, x_test = x_train.astype(np.float32) / 255., x_test.astype(np.float32) / 255.

x_train, x_test = np.expand_dims(x_train, axis=3), np.expand_dims(x_test, axis=3)

y_train_ohe = tf.one_hot(y_train, depth=10).numpy()
y_test_ohe = tf.one_hot(y_test, depth=10).numpy()


def conv3(no_filters, stride=1, kernel_size=(3, 3)):
    return Conv2D(filters=no_filters, kernel_size=kernel_size, strides=stride, padding='same')


class ResnetBlock(keras.Model):

    def __init__(self, channels, **kwargs):
        super(ResnetBlock, self).__init__(**kwargs)
        self.no_channels = channels
        self.resconv = Conv2D(self.no_channels, kernel_size=(1, 1), use_bias=False)
        self.c1 = conv3(channels)
        self.bn1 = BatchNormalization()
        self.c2 = conv3(channels)
        self.bn2 = BatchNormalization()

    def call(self, inputs):
        residual = inputs
        residual = self.resconv(residual)
        x = self.c1(residual)
        x = tf.nn.relu(x)
        x = self.bn1(x)
        x = self.c2(x)
        x = tf.nn.relu(x)
        x = self.bn2(x)
        x = residual + x

        return x


class ResNet(keras.Model):
    def __init__(self, init_channels, num_classes, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.next_channels = init_channels
        self.conv_init = conv3(self.next_channels)
        self.block1 = ResnetBlock(self.next_channels)
        self.next_channels *= 2
        self.block2 = ResnetBlock(self.next_channels)
        self.next_channels *= 2
        self.block3 = ResnetBlock(self.next_channels)
        self.bn = BatchNormalization()
        self.gap = GlobalAveragePooling2D()
        self.d = Dense(num_classes)

    def call(self, inputs):
        x = self.conv_init(inputs)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.bn(x)
        x = self.gap(x)
        x = self.d(x)
        return x


def main():
    num_classes = 10
    batch_size = 32
    epochs = 15

    model = ResNet(32, num_classes)
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    model.build(input_shape=(None, 28, 28, 1))
    print("Number of variables in the model : ", len(model.variables))

    model.summary()
    model.fit(x_train, y_train_ohe, batch_size=batch_size, epochs=epochs,
              validation_data=(x_test, y_test_ohe), verbose=1)

    model.save(r'C:\Users\Wojtek\Documents\Projects\DeepLearningPlayground\Resnet\model', save_format='tf')

    scores = model.evaluate(x_test, y_test_ohe, batch_size, verbose=1)
    print("Final test loss and accuracy :", scores)


if __name__ == '__main__':
    main()
