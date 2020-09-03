from absl import app
from absl import flags
from tensorflow import keras

import models
from utils import load_img

keras.backend.clear_session()

flags.DEFINE_string(
    'content_path', r'C:\Users\Wojtek\Documents\Projects\DeepLearningPlayground\NeuralStyleTransfer\ola.jpg',
    'Path to content image.')
flags.DEFINE_string(
    'style_path', r'C:\Users\Wojtek\Documents\Projects\DeepLearningPlayground\NeuralStyleTransfer\style1.jpg',
    'Path to style image.')
flags.DEFINE_string('output_dir', '.', 'Output directory.')
flags.DEFINE_integer('epochs', 20, 'Epochs to train.')
flags.DEFINE_integer('steps_per_epoch', 500, 'Steps per epoch.')
flags.DEFINE_float('tv_weight', 1e8, 'Total variation weight.')
flags.DEFINE_float('content_weight', 10000, 'Content weight.')
flags.DEFINE_float('style_weight', 0.10, 'Style weight.')
flags.DEFINE_float('learning_rate', 0.02, 'Learning rate.')
flags.DEFINE_float('beta_1', 0.99, 'Beta 1.')
flags.DEFINE_float('beta_2', 0.999, 'Beta 2.')
flags.DEFINE_float('epsilon', 0.10, 'Epsilon.')
flags.DEFINE_float('max_dim', 512, 'Max dimension to crop I/O image.')
flags.mark_flags_as_required(['content_path', 'style_path'])
FLAGS = flags.FLAGS


def main(argv):
    del argv

    content_img = load_img(FLAGS.content_path)
    style_img = load_img(FLAGS.style_path)

    mdl = models.StyleContent(content_img, style_img)
    mdl.train()


if __name__ == '__main__':
    app.run(main)
