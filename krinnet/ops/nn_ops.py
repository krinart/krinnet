import contextlib

import tensorflow as tf


@contextlib.contextmanager
def scope(name_or_scope):
    if name_or_scope:
        with tf.variable_scope(name_or_scope):
            yield

    else:
        yield


def fully_connected(input_tensor, layer_size, name_or_scope=None, activation=None, weights=None,
                    bias=None, random_state=None):

    with scope(name_or_scope):

        if weights is None:
            weights = tf.get_variable(
                'W', shape=(input_tensor.shape[1], layer_size),
                initializer=tf.truncated_normal_initializer(seed=random_state))

        if bias is None:
            bias = tf.get_variable('b', shape=(layer_size,), initializer=tf.constant_initializer(0))

        output = tf.nn.bias_add(tf.matmul(input_tensor, weights), bias)

        if activation:
            output = activation(output)

        return output
