import numpy as np
import tensorflow as tf

from krinnet import context
from krinnet import executor as krn_executor
from krinnet import layers
from krinnet import utils


class BaseNetwork(object):
    def __init__(self, layers, name=None):
        assert len(layers) > 0, 'Layers con not be empty'

        self.name = name

        # Hidden layers, but not necessarily all of them.
        # In case of AutoEncoder only encoding layers, without decoding.
        # Basically only layers with weights/biases for train.
        self.layers = layers

        # To be populated at a building stage
        self.input_layer = None
        self.all_layers = None  # includes input/hidden/output/loss layers

        self.loss_tensor = None
        self.output_tensor = None

        # Summaries accumulators
        self.train_summaries = []
        self.test_summaries = []
        self.stat_summaries = []

        # This one is probably a poor decision, but to make life more comfortable
        # it is good to save executor with full context (session, graph) in a net
        self.executor = None
        self.minimizer = None

    def get_train_feed(self, context):
        feed_dict = {}

        for layer in self.all_layers:
            feed_dict.update(layer.get_train_feed(context) or {})

        return feed_dict

    def get_test_feed(self, context):
        feed_dict = {}

        for layer in self.all_layers:
            feed_dict.update(layer.get_test_feed(context) or {})

        return feed_dict

    def build_loss(self, output_tensor):
        raise NotImplementedError

    def build_extra_layers(self, current_tensor):
        return current_tensor, []

    def build_input_layer(self, input_shape):
        if isinstance(self.layers[0], layers.BaseInputLayer):
            input_layer = self.layers.pop(0)
        else:
            input_layer = layers.InputLayer()

        current_tensor = input_layer.build(input_shape)

        return input_layer, current_tensor

    def build_hidden_layers(self, current_tensor):
        for layer_i, layer in enumerate(self.layers):
            current_tensor = layer.build_and_apply(current_tensor, layer_i=layer_i)

        return current_tensor

    def set_executor(self, executor):
        self.executor = executor

    def build(self, input_shape, optimizer=None, restore=False):
        self.executor = krn_executor.Executor()

        with self.executor.context:
            self.build_layers(input_shape)

            with tf.variable_scope('optimizer'):
                self.minimizer = optimizer and optimizer.minimize(self.loss_tensor)

        if restore:
            self.restore_model()
        else:
            self.executor.initialize()

    def build_layers(self, input_shape):
        self.input_layer, current_tensor = self.build_input_layer(input_shape)

        current_tensor = self.build_hidden_layers(current_tensor)

        current_tensor, extra_layers = self.build_extra_layers(current_tensor)

        self.output_tensor = current_tensor

        self.loss_tensor, loss_layer = self.build_loss(current_tensor)

        self.all_layers = [self.input_layer,  loss_layer] + self.layers + extra_layers

        for layer in self.all_layers:
            self.train_summaries.extend(layer.train_summaries)
            self.test_summaries.extend(layer.test_summaries)
            self.stat_summaries.extend(layer.stat_summaries)

    def get_loss_tensor(self):
        return self.loss_tensor

    def _run_test(self, fetches, X=None, Y=None, **extra_feed_dict):
        run_context = context.Context(X=X, Y=Y)
        feed_dict = self.get_test_feed(run_context)
        feed_dict.update(extra_feed_dict)
        return self.executor.run(fetches, feed_dict=feed_dict)

    def _run_train(self, fetches, X=None, Y=None):
        run_context = context.Context(X=X, Y=Y)
        return self.executor.run(fetches, feed_dict=self.get_train_feed(run_context))

    def train_step(self, X, Y=None):
        assert self.minimizer, 'minimizer is not set'
        return self._run_train(self.minimizer, X=X, Y=Y)

    def save_model(self, path=None):
        path = path or (self.name and 'models/{}'.format(self.name))
        assert path is not None, 'path is empty'

        path = utils.verify_path_is_empty(path)

        with self.executor.graph.as_default():
            saver = tf.train.Saver()
            saver.save(self.executor.session, path + '/model.ckpt')

    def restore_model(self, path=None):
        path = path or (self.name and 'models/{}'.format(self.name))
        assert path is not None, 'path is empty'

        with self.executor.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self.executor.session, path + '/model.ckpt')


class ClassifierNetwork(BaseNetwork):
    def __init__(self, layers, multiclass=True, *args, **kwargs):
        self.multiclass = multiclass
        super(ClassifierNetwork, self).__init__(layers, *args, **kwargs)
    
    def build_loss(self, output_tensor):
        loss_layer = layers.ClassifierLossLayer(
            'softmax_cross_entropy' if self.multiclass else 'sigmoid_cross_entropy')
        loss_tensor = loss_layer.build(output_tensor, self)

        return loss_tensor, loss_layer


# TODO: this network should know about image size
class AutoEncoder(BaseNetwork):
    def __init__(self, layers, *args, **kwargs):
        self.encoded_tensor = None  # actual encoded input
        super(AutoEncoder, self).__init__(layers, *args, **kwargs)

    def build_extra_layers(self, current_tensor):
        self.encoded_tensor = current_tensor

        for layer in self.layers[::-1]:
            current_tensor = layer.apply_reverse(current_tensor)

        return current_tensor, []

    def build_loss(self, output_tensor):
        loss_layer = layers.AutoEncoderLossLayer()
        loss_tensor = loss_layer.build(output_tensor, self)
        return loss_tensor, loss_layer

    def encode(self, *images):
        return self._run_test(self.encoded_tensor, X=np.stack(images))

    def decode(self, *tensors):
        shape = self.encoded_tensor.shape.as_list()[1:]

        return self.executor.run(
            self.output_tensor,
            feed_dict={
                self.encoded_tensor: np.stack(t.reshape(shape) for t in tensors),
            })

    def transform(self, source, target, steps=10):
        # First encode each image into latent vectors
        encoded_src, encoded_tgt = self.encode(source.reshape(-1), target.reshape(-1))

        # Get transformations for each step
        transformations = utils.linear_image_transform(encoded_src, encoded_tgt, steps=steps)

        # Decode each step
        decoded_transofrmations = self.decode(*transformations)

        return decoded_transofrmations.reshape(-1, 28, 28).clip(min=0, max=1)

    def transform_to_gif(self, image_name, source, target, steps=10):
        import imageio

        images = self.transform(source, target, steps=steps)

        imageio.mimsave(image_name, images)


class SimpleGenerativeNetwork(BaseNetwork):

    def __init__(self, net_layers):
        super(SimpleGenerativeNetwork, self).__init__(net_layers)

        assert isinstance(self.layers[0], layers.GenerativeInputLayer), (
            'Input layer for SimpleGenerativeNetwork should be '
            'instance of layers.GenerativeInputLayer')

    def generate(self, latent_vector):
        self._run_test()
