import tensorflow as tf
from .layers import InputBlock, DownsampleBlock, BottleneckBlock, UpsampleBlock, OutputBlock


class Unet(tf.keras.Model):

    def __init__(self):
        super().__init__(self)

        self.input_block = InputBlock(filters=64)

        self.down_blocks = [DownsampleBlock(filters, idx)
                            for idx, filters in enumerate([64, 256, 512])]

        self.bottleneck_block = BottleneckBlock(filters=1024)

        self.up_blocks = [UpsampleBlock(filters, idx)
                          for idx, filters in enumerate([512, 256, 64])]

        self.output_block = OutputBlock(filters=64, n_classes=2)

    def call(self, x, training=True):

        skip_connections = []
        out, res = self.input_block(x)
        skip_connections.append(res)

        for d_block in self.down_blocks:
            out, res = d_block(out)
            skip_connections.append(res)

        out = self.bottleneck_block(out)

        for up_block in self.up_blocks:
            out = up_block(out, skip_connections.pop())

        out = self.output_block(out), skip_connections.pop()

        return out
