import tensorflow as tf


class BaseSummary(object):
    def __init__(self, name, obj):
        self.name = name
        self.obj = obj

        # from IPython.core.debugger import Tracer;  Tracer()()

        self.placeholder = tf.placeholder(
            obj.dtype, shape=obj.shape, name='{}_smr_plhdr'.format('temp_name'))

        self.summary = self.register_summary()

    def register_summary(self):
        raise NotImplementedError

    def _as_graph_element(self):
        # Internal tf black magic
        conv_fn = getattr(self.obj, "_as_graph_element", None)
        if conv_fn and callable(conv_fn):
            return conv_fn()
        return self.obj

    def calculate(self, value):
        raise NotImplementedError


class MontageImagesSummary(BaseSummary):
    def register_summary(self):

        assert False, self.obj

        return tf.summary.image(self.name, self.placeholder)

    def calculate(self, value):
        return value.reshape(-1, 28, 28, 1)
