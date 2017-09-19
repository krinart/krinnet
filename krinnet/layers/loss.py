import tensorflow as tf

from krinnet import utils
from krinnet import layers


class ClassifierLossLayer(layers.BaseLossLayer):
    def __init__(self, activation, layer_name='loss_layer'):
        if activation == "sigmoid_cross_entropy":
            self.activation = tf.nn.sigmoid_cross_entropy_with_logits
        elif activation == "softmax_cross_entropy":
            self.activation = tf.nn.softmax_cross_entropy_with_logits
        else:
            raise ValueError('Unknown activation: {}'.format(activation))

        self.labels_placeholder = None  # placeholder for actual labels

        super(ClassifierLossLayer, self).__init__(layer_name=layer_name)

    def build(self, input_tensor, net):
        with self.scope():
            input_tensor = utils.ensure_tensor_dimensionality(input_tensor, 2)

            self.labels_placeholder = tf.placeholder(
                tf.float32, shape=(None, int(input_tensor.shape[1])), name='Y')

            loss = tf.reduce_mean(
                self.activation(labels=self.labels_placeholder, logits=input_tensor))

            for layer in net.layers:
                loss = layer.contribute_to_loss(loss)

            output = tf.identity(loss, 'loss')

            self.register_train_summary(tf.summary.scalar('train_loss', loss))
            self.register_test_summary(tf.summary.scalar('test_loss', loss))

        with self.scope('accuracy_layer'):
            correct_prediction = tf.equal(
                tf.argmax(self.labels_placeholder, 1), tf.argmax(input_tensor, 1))

            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

            self.register_train_summary(
                tf.summary.scalar('train_accuracy', accuracy))
            self.register_test_summary(tf.summary.scalar('test_accuracy', accuracy))

        return output

    def get_train_feed(self, context):
        return {self.labels_placeholder: context.Y}

    def get_test_feed(self, context):
        if context.Y is not None:
            return {self.labels_placeholder: context.Y}


class AutoEncoderLossLayer(layers.BaseLossLayer):
    def __init__(self, layer_name='loss_layer'):
        super(AutoEncoderLossLayer, self).__init__(layer_name=layer_name)

    def build(self, input_tensor, net):
        with self.scope():
            input_tensor = utils.ensure_tensor_dimensionality(input_tensor, 2)

            loss = tf.reduce_mean(
                tf.squared_difference(net.input_layer.input_placeholder, input_tensor), name='loss')

            for layer in net.layers:
                loss = layer.contribute_to_loss(loss)

            output = tf.identity(loss, 'loss')

            self.register_train_summary(tf.summary.scalar('train_loss', loss))
            self.register_test_summary(tf.summary.scalar('test_loss', loss))

            self.register_test_summary(
                tf.summary.image(
                    'input', tf.reshape(input_tensor, [-1, 28, 28, 1])))

        return output
