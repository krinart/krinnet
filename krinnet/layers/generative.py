import tensorflow as tf

from krinnet import utils
from krinnet.layers import base


class GenerativeInputLayer(base.BaseInputLayer):
    def __init__(self, size, c_dim=1, z_dim=8, scale=8):
        self.x_size, self.y_size = size
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.scale = scale

        self.z_placeholder = None
        super(GenerativeInputLayer, self).__init__(layer_name='input_layer')

    def build(self, training_data, training_labels):
        with self.scope():
            self.z_placeholder = tf.placeholder(tf.float32, shape=(1, self.z_dim), name='z')

            x = tf.linspace(-10, 10, self.x_size)
            y = tf.linspace(-10, 10, self.y_size)
            r = tf.sqrt(x**2 + y**2)


    def get_train_feed(self, context):
        pass
