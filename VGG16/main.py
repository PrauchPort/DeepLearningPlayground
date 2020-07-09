import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers
import argparse
import numpy as np


from network import VGG16

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
argparser = argparse.ArgumentParser()


argparser.add_argument('--train_dir', type=str, default='/tmp/cifar10_train',
                       help="Directory where to write event logs and checkpoint.")
argparser.add_argument('--max_steps', type=int, default=1000000,
                       help="""Number of batches to run.""")
argparser.add_argument('--log_device_placement', action='store_true',
                       help="Whether to log device placement.")
argparser.add_argument('--log_frequency', type=int, default=10,
                       help="How often to log results to the console.")


def normalize(X_train, X_test):
    X_train = X_train / 255.
    X_test = X_test / 255.

    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    print('mean:', mean, 'std:', std)
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)
    return X_train, X_test


def prepare_cifar(x, y):

    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.int32)
    return x, y


def compute_loss(logits, labels):
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    )


def main():

    tf.random.set_seed(22)

    print('loading data...')

    (x, y), (x_test, y_test) = datasets.cifar10.load_data()
    x, x_test = normalize(x, x_test)

    print(x.shape, y.shape, x_test.shape, y_test.shape)

    train_loader = tf.data.Dataset.from_tensor_slices((x, y))
    train_loader = train_loader.map(prepare_cifar).shuffle(50000).batch(256)

    test_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_loader = test_loader.map(prepare_cifar).shuffle(10000).batch(256)
    print('done.')

    model = VGG16([32, 32, 3])

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.build(input_shape=(None, 32, 32, 3))

    print("Number of variables in the model : ", len(model.variables))

    model.summary()
    model.fit(train_loader, epochs=40,
              validation_data=test_loader, verbose=1)


if __name__ == '__main__':
    main()
