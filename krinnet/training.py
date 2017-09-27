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

    X_train, X_test, Y_train, Y_test = utils.train_test_split(
        X, Y, test_size=test_size, use_examples_num=use_examples_num,
        random_state=random_state)

    train_state = TrainState(net)

    reporter = net.name and reporting.Reporter(
        net, X_train, X_test, Y_train=Y_train, Y_test=Y_test,
        summary_logdir='logs/{}'.format(net.name),
        summary_step=summary_step,
        print_accuracy_step=print_accuracy_step,
        print_error_step=print_error_step,
    )

    with utils.catch_keyboard_interrupted():
        for _ in tqdm(range(epochs)):
            batches = utils.batch_iterator(X_train, Y_train, batch_size, random_state=random_state)

            for X_batch, Y_batch in batches:
                train_state.train_step(X=X_batch, Y=Y_batch)

                reporter and reporter.step(train_state.global_step)

    train_state.save()
    reporter and reporter.finalize()

    return train_state


class TrainState(object):
    def __init__(self, net):
        self.net = net
        self.executor = krn_executor.Executor()

        self.global_step = 0

        with self.executor.context:
            self._initialize()

        self.restore()

    def _initialize(self):
        self.global_step_tensor = tf.get_variable('global_step', shape=(), dtype=tf.int64)
        self.executor.initialize()

    def train_step(self, X, Y=None):
        self.net.train_step(X=X, Y=Y)

        self.global_step += 1

    def restore(self):
        if not self.net.name or not utils.is_interactive():
            return

        # TODO: this is horrible
        if not os.path.exists('models/{}'.format(self.net.name)):
            return

        if input('Restore state? (n): ').lower() in ['y', 'yes']:
            path = self.net.restore_model()

            self.executor.restore_model(path, model_name='train_state')
            self.global_step = self.global_step_tensor.eval(session=self.executor.session)

    def save(self):
        if not self.net.name or not utils.is_interactive():
            return

        response = input('Finished. Save state? (n): ') or 'n'

        if response.lower() in ['y', 'yes']:
            path = self.net.save_model(force=True)

            self.executor.run(self.global_step_tensor.assign(self.global_step))
            self.executor.save_model(path, model_name='train_state', force=True)
