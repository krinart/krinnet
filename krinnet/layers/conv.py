import tensorflow as tf

from krinnet import utils
from krinnet.layers import base


class Conv2DLayer(base.BaseLayer):
    default_w_initializer = tf.truncated_normal_initializer()
    default_b_initializer = .1

    def __init__(self, filter_size, strides=(1, 1, 1, 1), activation=None, layer_name=None,
                 weights_initializer=None, bias_initializer=None, dropout_keep_prob=None,
                 weights_decay=None, random_state=None):
        self.filter_size = filter_size
        self.strides = strides
        self.activation = activation
        self.random_state = random_state

        self.weights_initializer = utils.initialize_variable(
            weights_initializer, self.default_w_initializer, random_state=random_state)
        self.bias_initializer = utils.initialize_variable(
            bias_initializer, self.default_b_initializer, random_state=random_state)

        self.dropout_keep_prob = dropout_keep_prob
        self.weights_decay = weights_decay

        self.input_layer_size = None
        self.weights = None
        self.bias = None
        self.output = None
        self.dropout = None
        self.input_shape = None
        super(Conv2DLayer, self).__init__(layer_name=layer_name)

    def build(self, layer_i, input_tensor):
        self.layer_name = self.layer_name or 'conv2d_layer_{}'.format(layer_i)

        with self.scope():
            input_tensor = self.ensure_input_tensor_dimensionality(input_tensor, 4)

            self.weights = tf.get_variable(
                'W', shape=self.filter_size,
                dtype=tf.float32,
                initializer=self.weights_initializer)

            self.bias = tf.get_variable(
                'b', shape=self.filter_size[-1], dtype=tf.float32,
                initializer=self.bias_initializer)

            self.input_shape = tf.stack([tf.shape(input_tensor)[0]] + input_tensor.shape.as_list()[1:])
            output = tf.nn.conv2d(
                input_tensor, self.weights, strides=self.strides, padding='SAME')

            output = tf.nn.bias_add(output, self.bias)

            if self.activation:
                output = self.activation(output)

            before_dropout = output

            if self.dropout_keep_prob:
                self.dropout = tf.placeholder(tf.float32, shape=(), name='dropout_keep_prob')
                output = tf.nn.dropout(output, self.dropout)

            self.register_stat_summary(tf.summary.histogram('weights', self.weights))
            self.register_stat_summary(tf.summary.histogram('bias', self.bias))

            # self.register_train_summary(
            #     executor.Executor.histogram_summary('train_output', before_dropout))
            # self.register_test_summary(
            #     executor.Executor.histogram_summary('test_output', before_dropout))

        return output

    def apply_reverse(self, input_tensor):
        if self.weights is None or self.bias is None:
            raise RuntimeError('Layer is not built')

        with self.reverse_scope():
            input_tensor = utils.ensure_tensor_dimensionality(input_tensor, 4)

            output = tf.nn.bias_add(input_tensor, tf.negative(self.bias))

            output = tf.nn.conv2d_transpose(output, self.weights, output_shape=self.input_shape,
                                            strides=self.strides, padding='SAME')

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
