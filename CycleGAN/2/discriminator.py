import tensorflow as tf
import tensorflow.keras as keras
import ops

class Discriminator(keras.Model):
    def __init__(self, scope: str='Discriminator', reg: float=0.0005, norm:str="instance"):
        super(Discriminator, self).__init__(name=scope)
        self.ck1 = ops.Ck