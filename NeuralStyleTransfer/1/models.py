import tensorflow as tf
from absl import flags

from tensorflow.python.keras import applications
from tensorflow.python.keras import models

from tqdm.autonotebook import tqdm

from ops import get_style_content_loss, calculate_gram_matrix, get_vgg_layers

from utils import get_terminal_width, save_img

FLAGS = flags.FLAGS


class StyleContent(models.Model):
    STYLE_LAYERS = [
        'block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'
    ]

    CONTENT_LAYERS = ['block5_conv2']

    def __init__(self, content_img, style_img):
        super(StyleContent, self).__init__()

        self.content_img = content_img
        self.style_img = style_img
        self.img = tf.Variable(content_img)
        self.vgg = get_vgg_layers(self.STYLE_LAYERS + self.CONTENT_LAYERS)
        self.num_style_layers = len(self.STYLE_LAYERS)

        self.vgg.trainable = False
        self.style_targets = self(self.style_img)['style']
        self.content_targets = self(self.content_img)['content']
        self.opt = tf.optimizers.Adam(learning_rate=FLAGS.learning_rate,
                                      beta_1=FLAGS.beta_1, beta_2=FLAGS.beta_2, epsilon=FLAGS.epsilon)

    def call(self, inputs, **kwargs):
        inputs = inputs * 255.0
        preprocessed_input = applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [
            calculate_gram_matrix(style_output) for style_output in style_outputs
        ]

        content = {
            content_name: value
            for content_name, value in zip(self.CONTENT_LAYERS, content_outputs)
        }

        style = {
            content_name: value
            for content_name, value in zip(self.STYLE_LAYERS, style_outputs)
        }

        return {'content': content, 'style': style}

    def train(self):
        for n in range(FLAGS.epochs):
            for _ in tqdm(
                iterable=range(FLAGS.steps_per_epoch),
                ncols=int(get_terminal_width() * .9),
                desc=tqdm.write(f'Epoch {n + 1} / {FLAGS.epochs}'),
                unit=' steps'
            ):
                self.train_step(self.img)
            save_img(self.img.read_value(), n+1)

    @tf.function()
    def train_step(self, img):
        with tf.GradientTape() as tape:
            outputs = self(img)
            loss = get_style_content_loss(outputs, self.content_targets,
                                          self.style_targets, self.CONTENT_LAYERS,
                                          self.CONTENT_LAYERS, img)

        grad = tape.gradient(loss, img)
        self.opt.apply_gradients([(grad, img)])
        img.assign(tf.clip_by_value(img, 0.0, 1.0))

    def compute_output_signature(self, input_signature):
        pass
