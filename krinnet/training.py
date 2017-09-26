import os

import tensorflow as tf
from tqdm import tqdm

from krinnet import executor as krn_executor
from krinnet import reporting
from krinnet import utils


def train_nn(net, X, Y, epochs, learning_rate, optimizer_cls=tf.train.AdamOptimizer, batch_size=512,
             random_state=None, summary_step=5, test_size=.2, use_examples_num=None,
             print_accuracy_step=None, print_error_step=None):

    net.build(input_shape=X.shape, optimizer=optimizer_cls(learning_rate))
    maybe_restore_model(net)

    X_train, X_test, Y_train, Y_test = utils.train_test_split(
        X, Y, test_size=test_size, use_examples_num=use_examples_num,
        random_state=random_state)

    reporter = net.name and reporting.Reporter(
        net, X_train, X_test, Y_train=Y_train, Y_test=Y_test,
        summary_logdir='logs/{}'.format(net.name),
        summary_step=summary_step,
        print_accuracy_step=print_accuracy_step,
        print_error_step=print_error_step,
    )

    train_state = TrainState(net)

    with utils.catch_keyboard_interrupted():
        for _ in tqdm(range(epochs)):
            batches = utils.batch_iterator(X_train, Y_train, batch_size, random_state=random_state)

            for X_batch, Y_batch in batches:
                train_state.train_step(X=X_batch, Y=Y_batch)

                if reporter:
                    reporter.step(train_state.global_step)

    net.executor.finalize()
    maybe_save_model(net)


class TrainState(object):
    def __init__(self, net):
        self.net = net
        self.executor = krn_executor.Executor()

        self.global_step = 0

        with self.executor.context:
            self._initialize()

    def _initialize(self):
        self.global_step_tensor = tf.get_variable('global_step', shape=())

    def train_step(self, X, Y=None):
        self.net.train_step(X=X, Y=Y)

        self.global_step += 1

    def restore(self):
        pass

    def save(self):
        pass


def maybe_restore_model(net):
    if not net.name or not utils.is_interactive() or not os.path.exists('models/{}'.format(net.name)):
        return

    if input('Restore model? (n): ').lower() in ['y', 'yes']:
        net.restore_model()


def maybe_save_model(net):
    if not net.name or not utils.is_interactive():
        return

    response = input('Finished. Save model? (n): ') or 'n'

    if response.lower() in ['y', 'yes']:
        net.save_model()
