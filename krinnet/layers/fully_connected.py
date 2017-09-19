import tensorflow as tf

from krinnet import utils
from krinnet import ops
from krinnet.layers import base


class FullyConnectedLayer(base.BaseHiddenLayer):
    default_w_initializer = tf.contrib.layers.xavier_initializer
    default_b_initializer = .1

    layer_basename = 'fc_layer'
    n_dimensions = 2

    def __init__(self, layer_size, activation=None, layer_name=None,
                 weights_initializer=None, bias_initializer=None, dropout_keep_prob=None,
                 weights_decay=None, random_state=None):
        self.layer_size = layer_size
        self.activation = activation

        self.weights_initializer = utils.initialize_variable(
            weights_initializer, self.default_w_initializer, random_state=random_state)
        self.bias_initializer = utils.initialize_variable(
            bias_initializer, self.default_b_initializer, random_state=random_state)

        self.dropout_keep_prob = dropout_keep_prob
        self.weights_decay = weights_decay

        self.weights = None
        self.bias = None
        self.output = None
        self.dropout = None
        super(FullyConnectedLayer, self).__init__(layer_name=layer_name)

    def build(self, input_tensor, layer_i=None):
        input_tensor = self.build_input_tensor_dimensionality(input_tensor)

        self.weights = tf.get_variable(
            'W', shape=(input_tensor.shape[1], self.layer_size),
            dtype=tf.float32,
            initializer=self.weights_initializer)

        self.bias = tf.get_variable(
            'b', shape=self.layer_size, dtype=tf.float32,
            initializer=self.bias_initializer)

    def apply(self, input_tensor):
        input_tensor = self.verify_input_tensor_dimensionality(input_tensor)

        output = ops.fully_connected(
            input_tensor, self.layer_size, activation=self.activation,
            weights=self.weights, bias=self.bias)

        before_dropout = output

        if self.dropout_keep_prob:
            self.dropout = tf.placeholder(tf.float32, shape=(), name='dropout_keep_prob')
            output = tf.nn.dropout(output, self.dropout)

        self.register_stat_summary(tf.summary.histogram('weights', self.weights))
        self.register_stat_summary(tf.summary.histogram('bias', self.bias))

        if False:  # TODO
            self.register_train_summary(tf.summary.histogram('train_output', before_dropout))
            self.register_test_summary(tf.summary.histogram('test_output', before_dropout))

        return output

    def apply_reverse(self, input_tensor):
        if self.weights is None or self.bias is None:
            raise RuntimeError('Layer is not built')

        with self.reverse_scope():
            input_tensor = utils.ensure_tensor_dimensionality(input_tensor, 2)

            output = tf.nn.bias_add(input_tensor, tf.negative(self.bias))

            output = tf.matmul(output, self.weights, transpose_b=True)

            if self.activation:
                output = self.activation(output)

            return self.restore_dimensionality_after_reverse(output)

    def contribute_to_loss(self, loss):
        if self.weights_decay is not None:
            loss += self.weights_decay * tf.nn.l2_loss(self.weights)

        return loss

    def get_train_feed(self, context):
        if self.dropout is not None and self.dropout_keep_prob:
            return {self.dropout: self.dropout_keep_prob}

    def get_test_feed(self, context):
        if self.dropout is not None:
            return {self.dropout: 1}
