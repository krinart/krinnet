import tensorflow as tf

from tqdm import tqdm

from krinnet import reporting
from krinnet import utils
from krinnet import storage


def train_nn(net, X, Y, epochs, learning_rate, optimizer_cls=tf.train.AdamOptimizer, batch_size=512,
             random_state=None, summary_step=5, test_size=.2, use_examples_num=None,
             print_accuracy_step=None, print_error_step=None):

    net.build(input_shape=X.shape, optimizer=optimizer_cls(learning_rate))

    X_train, X_test, Y_train, Y_test = utils.train_test_split(
        X, Y, test_size=test_size, use_examples_num=use_examples_num,
        random_state=random_state)

    train_state = TrainState(net, X_train, X_test, Y_train, Y_test)

    reporter = reporting.Reporter(
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
    reporter.finalize()

    return train_state


class TrainState(object):

    def __init__(self, net, X_train, X_test, Y_train, Y_test):
        self.net = net
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test

        self.global_step = 0
        self.storage = storage.VarStorage('{}_train_state'.format(net.name))

        self.restore()

    def train_step(self, X, Y=None):
        self.net.train_step(X=X, Y=Y)
        self.global_step += 1

    def restore(self):
        if not self.net.name or not utils.is_interactive():
            return

        try:
            restored_state = self.storage.restore_values({'global_step': tf.int32})
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
            self.storage.save_values({'global_step': (int(self.global_step), tf.int32)})
