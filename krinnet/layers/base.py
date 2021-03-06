import contextlib

import tensorflow as tf

from krinnet import utils


@contextlib.contextmanager
def dummy_scope():
    yield


class BaseLayer(object):

    def __init__(self, layer_name=None):
        self.layer_name = layer_name

        self.train_summaries = []
        self.test_summaries = []
        self.stat_summaries = []

    def scope(self, layer_name=None):
        layer_name = layer_name or self.layer_name
        return tf.variable_scope(layer_name) if layer_name else dummy_scope

    # TODO: maybe move next 2 methods into mixin class
    def reverse_scope(self):
        assert self.layer_name, 'layer_name is not specified'
        return tf.variable_scope('{}_reversed'.format(self.layer_name))

    def apply_reverse(self, input_tensor):
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


class BuildableLayer(BaseLayer):

    def build_name(self, *args, **kwargs):
        pass

    def build(self, *args, **kwargs):
        raise NotImplementedError


class AppliableLayer(BuildableLayer):
    def apply(self, input_tensor):
        raise NotImplementedError

    def build_and_apply(self, input_tensor, layer_i=None):
        self.build_name(layer_i=layer_i)

        with self.scope():
            self.build(input_shape=input_tensor.shape.as_list())
            return self.apply(input_tensor)


class BaseInputLayer(BuildableLayer):
    def build(self, data_shape):
        raise NotImplementedError


class BaseHiddenLayer(AppliableLayer):

    layer_basename = None
    n_dimensions = None

    def __init__(self, *args, **kwargs):
        super(BaseHiddenLayer, self).__init__(*args, **kwargs)

        self.input_tensor_shape = None       # Has -1 for wildcard dimension
        self.input_tensor_shape_list = None  # Has None for wildcard dimension

    def build_name(self, layer_i=None):
        self.layer_name = self.layer_name or '{}_{}'.format(self.layer_basename, layer_i)

    def build_input_tensor_dimensionality(self, shape):
        assert shape[0] is None, 'Shape[0] should be None, actual shape: {}'.format(shape)
        assert self.n_dimensions, 'n_dimensions is not set'

        if self.input_tensor_shape is not None:
            raise RuntimeError(
                'input_tensor_shape is already set for layer {}'.format(self.layer_name))

        shape = list(shape)
        self.input_tensor_shape_list = shape

        valid_shape = utils.cast_shape_to_dimensionality(shape, self.n_dimensions)
        self.input_tensor_shape = [(s or -1) for s in valid_shape]

        return valid_shape

    def verify_input_tensor_dimensionality(self, input_tensor):
        if self.input_tensor_shape_list != input_tensor.shape.as_list():
            raise ValueError(
                "Tensors' shape during building and applying are different: build={}, "
                "apply={}".format(self.input_tensor_shape_list, input_tensor.shape.as_list()))

        return utils.ensure_tensor_dimensionality(input_tensor, self.n_dimensions)

    def restore_dimensionality_after_reverse(self, tensor):
        if self.input_tensor_shape is None:
            raise RuntimeError(
                'input_tensor_shape has not been set for layer {}'.format(self.layer_name))

        return tf.reshape(tensor, self.input_tensor_shape)


class BaseLossLayer(BuildableLayer):
    def build(self, input_tensor, layers):
        raise NotImplementedError
