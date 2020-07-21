import tensorflow as tf
from tensorflow import keras


class RNNColorBot(keras.Model):
    """
    Multi-layer (LSTM) RNN that regresses on real-valued vector labels.
    """

    def __init__(self, rnn_cell_sizes, label_dimension, keep_prob):
        """Constructs an RNNColorbot.
        Args:
          rnn_cell_sizes: list of integers denoting the size of each LSTM cell in
            the RNN; rnn_cell_sizes[i] is the size of the i-th layer cell
          label_dimension: the length of the labels on which to regress
          keep_prob: (1 - dropout probability); dropout is applied to the outputs of
            each LSTM layer
        """
        super(RNNColorBot, self).__init__(name="")

        self.rnn_cell_sizes = rnn_cell_sizes
        self.label_dimension = label_dimension
        self.keep_prob = keep_prob

        self.cells = [keras.layers.LSTMCell(size) for size in rnn_cell_sizes]
        self.relu = keras.layers.Dense(label_dimension, activation=tf.nn.relu)

    def call(self, inputs, training=None):
        """
        Implements the RNN logic and prediction generation.
        Args:
          inputs: A tuple (chars, sequence_length), where chars is a batch of
            one-hot encoded color names represented as a Tensor with dimensions
            [batch_size, time_steps, 256] and sequence_length holds the length
            of each character sequence (color name) as a Tensor with dimension
            [batch_size].
          training: whether the invocation is happening during training
        Returns:
          A tensor of dimension [batch_size, label_dimension] that is produced by
          passing chars through a multi-layer RNN and applying a ReLU to the final
          hidden state.
        """
        (chars, sequence_length) = inputs

        chars = tf.transpose(chars, [1, 0, 2])
        batch_size = int(chars.shape[1])
        for l in range(len(self.cells)):
            cell = self.cells[l]
            outputs = []

            state = (tf.zeros((batch_size, self.rnn_cell_sizes[l])))
            chars = tf.unstack(chars, axis=0)

            for ch in chars:
                output, state = cell(ch, state)
                outputs.append(output)
            chars = tf.stack(outputs, axis=0)
            if training:
                chars = tf.nn.dropout(chars, self.keep_prob)

        batch_range = [i for i in range(batch_size)]

        indices = tf.stack([sequence_length - 1, batch_range], axis=1)

        hidden_states = tf.gather_nd(chars, indices)

        return self.relu(hidden_states)
