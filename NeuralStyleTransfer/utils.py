import os
import shutil

import tensorflow as tf
from absl import flags
from tensorflow import io

FLAGS = flags.FLAGS


def load_img(img_path):
    img = io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = FLAGS.max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape, antialias=True)
    img = img[tf.newaxis, :]
    return img


def save_img(image_batch, epoch):
    file_name = f'{epoch}.jpg'
    output_dir = os.path.join(FLAGS.output_dir, file_name)

    for i in image_batch:
        img = tf.image.encode_jpeg(tf.cast(i * 255, tf.uint8), format='rgb')
        tf.io.write_file(output_dir, img)


def get_terminal_width():
    width = shutil.get_terminal_size(fallback=(200, 24))[0]
    if width == 0:
        width = 120
    return width
