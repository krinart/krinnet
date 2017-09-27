import os

import tensorflow as tf
from tensorflow.python.framework import errors_impl as tf_errors
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

    train_state = TrainState(net, X_train, X_test, Y_train, Y_test)

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

    def __init__(self, net, X_train, X_test, Y_train, Y_test):
        self.net = net
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test

        self.global_step = 0
        self.var_store = VarStore('models/{}'.format(net.name), 'train_state')

        self.restore()

    def train_step(self, X, Y=None):
        self.net.train_step(X=X, Y=Y)
        self.global_step += 1

    def restore(self):
        if not self.net.name or not utils.is_interactive():
            return

        try:
            restored_state = self.var_store.restore_values({'global_step': tf.int32})
        except Exception as e:
            print(type(e))
            return

        print(restored_state)

        if 'global_step' in restored_state:
            self.global_step = restored_state['global_step']
            print('Restored global step: {}'.format(self.global_step))

        # # TODO: this is horrible
        # if not os.path.exists('models/{}'.format(self.net.name)):
        #     return
        #
        # if input('Restore state? (n): ').lower() in ['y', 'yes']:
        #     path = self.net.restore_model()

    def save(self):
        if not self.net.name or not utils.is_interactive():
            return

        response = input('Finished. Save state? (n): ') or 'n'

        if response.lower() in ['y', 'yes']:
            self.net.save_model(force=True)
            self.var_store.save_values({'global_step': (int(self.global_step), tf.int32)})


class VarStore(object):
    def __init__(self, path, name):
        self.path = path
        self.name = name

    def _save_value(self, name, value, dtype, executor):
        with executor.context():
            var = tf.get_variable('var_{}'.format(name), dtype=dtype, initializer=value)

            dimensions = tf.get_variable(
                'var_{}_dimensions'.format(name), dtype=tf.int32,
                initializer=len(var.shape))

            executor.initialize(var, dimensions)

            return tf.get_variable(
                'var_{}_shape'.format(name), dtype=tf.int32,
                initializer=var.shape)

    def _restore_value(self, name, dtype, executor):
        with executor.context():
            dimensions = tf.get_variable('var_{}_dimensions'.format(name), shape=(), dtype=tf.int32)
            executor.restore_model(self.path)

            shape = tf.get_variable(
                'var_{}_shape'.format(name),
                shape=executor.run(dimensions),
                dtype=tf.int32)
            executor.restore_model(self.path)

            var = tf.get_variable('var_{}'.format(name),
                                  shape=list(executor.run(shape)), dtype=dtype)
            executor.restore_model(self.path)

        return executor.run(var)

    def save_values(self, values_dict):
        executor = krn_executor.Executor(self.name)

        shapes = []
        for name, (value, dtype) in values_dict.items():
            shapes.append(self._save_value(name, value, dtype, executor))

        executor.initialize(*shapes)
        executor.save_model(self.path)

    def restore_values(self, values_dict):
        executor = krn_executor.Executor(self.name)

        result_dict = {}
        for name, dtype in values_dict.items():
            result_dict[name] = self._restore_value(name, dtype, executor)

        return result_dict
