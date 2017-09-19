from krinnet import context
from krinnet import layers


class BaseNetwork(object):
    def __init__(self, layers):
        assert len(layers) > 0, 'Layers con not be empty'

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

    def build_input_layer(self, X, Y):
        if isinstance(self.layers[0], layers.BaseInputLayer):
            input_layer = self.layers.pop(0)
        else:
            input_layer = layers.InputLayer()

        current_tensor = input_layer.build(X, Y)

        return input_layer, current_tensor

    def build_hidden_layers(self, current_tensor):
        for layer_i, layer in enumerate(self.layers):
            current_tensor = layer.build(layer_i, current_tensor)

        return current_tensor

    def set_executor(self, executor):
        self.executor = executor

    def build(self, X, Y):
        self.input_layer, current_tensor = self.build_input_layer(X, Y)

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

    def _run_test(self, fetches, X=None, Y=None):
        run_context = context.Context(X=X, Y=Y)
        return self.executor.run(fetches, feed_dict=self.get_test_feed(run_context))

    def _run_train(self, fetches, X=None, Y=None):
        run_context = context.Context(X=X, Y=Y)
        return self.executor.run(fetches, feed_dict=self.get_train_feed(run_context))

    def train_step(self, minimizer, X, Y=None):
        return self._run_train(minimizer, X=X, Y=Y)


class ClassifierNetwork(BaseNetwork):
    def __init__(self, layers, multiclass=True):
        self.multiclass = multiclass
        super(ClassifierNetwork, self).__init__(layers)
    
    def build_loss(self, output_tensor):
        loss_layer = layers.ClassifierLossLayer(
            'softmax_cross_entropy' if self.multiclass else 'sigmoid_cross_entropy')
        loss_tensor = loss_layer.build(output_tensor, self)

        return loss_tensor, loss_layer


class AutoEncoder(BaseNetwork):
    def __init__(self, layers):
        self.encoded_tensor = None  # actual encoded input
        super(AutoEncoder, self).__init__(layers)

    def build_extra_layers(self, current_tensor):
        self.encoded_tensor = current_tensor

        for layer in self.layers[::-1]:
            current_tensor = layer.apply_reverse(current_tensor)

        return current_tensor, []

    def build_loss(self, output_tensor):
        loss_layer = layers.AutoEncoderLossLayer()
        loss_tensor = loss_layer.build(output_tensor, self)
        return loss_tensor, loss_layer


class SimpleGenerativeNetwork(BaseNetwork):

    def __init__(self, net_layers):
        super(SimpleGenerativeNetwork, self).__init__(net_layers)

        assert isinstance(self.layers[0], layers.GenerativeInputLayer), (
            'Input layer for SimpleGenerativeNetwork should be '
            'instance of layers.GenerativeInputLayer')

    def generate(self, latent_vector):
        self._run_test()
