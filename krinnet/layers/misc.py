import tensorflow as tf

from krinnet import utils
from krinnet.layers import base


# TODO: introduce ImageInputLayer
class InputLayer(base.BaseInputLayer):
    def __init__(self):
        self.input_placeholder = None
        super(InputLayer, self).__init__(layer_name='input_layer')

    def build(self, input_shape):
        assert len(input_shape) > 1, 'Input shape should '

        input_shape = [None] + list(input_shape[1:])

        with self.scope():
            # TODO: handle dimensions more properly, currently dimensions after second simply lost
            self.input_placeholder = tf.placeholder(
                tf.float32, shape=input_shape, name='X')

        return self.input_placeholder

    def get_train_feed(self, context):
        return {self.input_placeholder: context.X}

    def get_test_feed(self, context):
        return {self.input_placeholder: context.X}


class Reshape(base.BaseHiddenLayer):
    layer_basename = 'reshape'

    def __init__(self, shape, layer_name=None):
        self.shape = shape
        self.original_shape = None
        super(Reshape, self).__init__(layer_name=layer_name)

    def build(self, *args, **kwargs):
        pass

    def apply(self, input_tensor):
        self.original_shape = input_tensor.shape.as_list()
        return tf.reshape(input_tensor, [-1] + list(self.shape))

    def apply_reverse(self, input_tensor):
        with self.reverse_scope():
            return tf.reshape(input_tensor, tf.stack([-1] + self.original_shape[1:]))


class MaxPoolLayer(base.BaseHiddenLayer):
    layer_basename = 'maxpool'

    def __init__(self, size_shortcut=None, size=None, layer_name=None):
        if size_shortcut:
            size = [1, size_shortcut, size_shortcut, 1]

        if not size:
            raise ValueError('size is not specified')

        self.size = size
        super(MaxPoolLayer, self).__init__(layer_name=layer_name)

    def build(self, *args, **kwargs):
        pass

    def apply(self, input_tensor):
        input_tensor = utils.ensure_tensor_dimensionality(input_tensor, 4)
        return tf.nn.max_pool(input_tensor, self.size, strides=[1, 2, 2, 1], padding='SAME')
