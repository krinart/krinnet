import tensorflow as tf

from krinnet import utils


class BaseLayer(object):
    def __init__(self, layer_name=None):
        self.layer_name = layer_name

        self.train_summaries = []
        self.test_summaries = []
        self.stat_summaries = []

        # store layer's input's shape in ensure_tensor_dimensionality
        # to restore it after .apply_reverse()
        self.input_tensor_shape = None

    def scope(self, layer_name=None):
        layer_name = layer_name or self.layer_name

        assert layer_name, 'layer_name is not specified'

        return tf.variable_scope(layer_name)

    # TODO: maybe move this method into mixin class along with .apply_reverse() method
    def reverse_scope(self):
        assert self.layer_name, 'layer_name is not specified'
        return tf.variable_scope('{}_reversed'.format(self.layer_name))

    def apply_reverse(self, input_tensor):
        raise NotImplementedError

    def build(self, *args, **kwargs):
        raise NotImplementedError

    def get_train_feed(self, context):
        pass

    def get_test_feed(self, context):
        pass

    def contribute_to_loss(self, loss):
        return loss

    def register_train_summary(self, summary):
        self.train_summaries.append(summary)

    def register_test_summary(self, summary):
        self.test_summaries.append(summary)

    def register_stat_summary(self, summary):
        self.stat_summaries.append(summary)

    def ensure_input_tensor_dimensionality(self, tensor, n_dimensions):
        if self.input_tensor_shape is not None:
            raise RuntimeError(
                'input_tensor_shape is already set for layer {}'.format(self.layer_name))

        self.input_tensor_shape = tf.shape(tensor)
        return utils.ensure_tensor_dimensionality(tensor, n_dimensions)

    def restore_dimensionality_after_reverse(self, tensor):
        if self.input_tensor_shape is None:
            raise RuntimeError(
                'input_tensor_shape has not been set for layer {}'.format(self.layer_name))

        return tf.reshape(tensor, self.input_tensor_shape)


class BaseLossLayer(BaseLayer):
    def build(self, input_tensor, layers):
        raise NotImplementedError


class BaseInputLayer(BaseLayer):
    def build(self, training_data, training_labels):
        raise NotImplementedError
