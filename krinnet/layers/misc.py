import tensorflow as tf

from krinnet import utils
from krinnet.layers import base


class InputLayer(base.BaseInputLayer):
    def __init__(self):
        self.input_placeholder = None
        super(InputLayer, self).__init__(layer_name='input_layer')

    def build(self, training_data, training_labels):

        with self.scope():
            self.input_placeholder = tf.placeholder(tf.float32, shape=(None, training_data.shape[1]), name='X')

        return self.input_placeholder

    def get_train_feed(self, context):
        return {self.input_placeholder: context.X}

    def get_test_feed(self, context):
        return {self.input_placeholder: context.X}


class Reshape(base.BaseLayer):
    def __init__(self, shape, layer_name=None):
        self.shape = shape
        self.original_shape = None
        super(Reshape, self).__init__(layer_name=layer_name)

    def build(self, layer_i, input_tensor):
        self.layer_name = self.layer_name or 'reshape_{}'.format(layer_i)

        with self.scope():
            self.original_shape = tf.shape(input_tensor)
            return tf.reshape(input_tensor, [-1] + list(self.shape))

    def apply_reverse(self, input_tensor):
        with self.reverse_scope():
            return tf.reshape(input_tensor, self.original_shape)


class MaxPoolLayer(base.BaseLayer):
    def __init__(self, size_shortcut=None, size=None, layer_name=None):
        if size_shortcut:
            size = [1, size_shortcut, size_shortcut, 1]

        if not size:
            raise ValueError('size is not specified')

        self.size = size
        super(MaxPoolLayer, self).__init__(layer_name=layer_name)

    def build(self, layer_i, input_tensor):
        self.layer_name = self.layer_name or 'max_pool_layer_{}'.format(layer_i)

        with self.scope():
            input_tensor = utils.ensure_tensor_dimensionality(input_tensor, 4)

            return tf.nn.max_pool(input_tensor, self.size, strides=[1, 2, 2, 1], padding='SAME')
