import os
import numpy as np
import tensorflow as tf

from tqdm import tqdm

from krinnet import reporting
from krinnet import utils
from krinnet import storage


def restore_train_state(net):
    if not utils.is_interactive() or not os.path.exists('models/{}'.format(net.name)):
        return 0

    try:
        state = storage.restore_values(
            '{}_train_state'.format(net.name),
            values_dict={
                'global_step': tf.int32})
    except storage.StorageError:
        return 0

    response = input('Restore state with global_step={}? (y)/n: '.format(state['global_step']))

    if not response in ['y', 'yes', '']:
        return 0

    net.restore_model()
    print('State restored')
    return state['global_step']


def save_train_state(net, global_step):
    if not utils.is_interactive():
        return

    response = input('Finished. Save state? (y)/n: ').lower() or 'n'

    if not response in ['y', 'yes', '']:
        return

    net.save_model(force=True)

    storage.save_values(
        '{}_train_state'.format(net.name),
        values_dict={'global_step': (int(global_step), tf.int32)})


def train_nn(net, X, Y, epochs, learning_rate, optimizer_cls=tf.train.AdamOptimizer, batch_size=512,
             random_state=None, summary_step=5, test_size=.2, use_examples_num=None,
             print_accuracy_step=None, print_error_step=None):

    net.build(input_shape=X.shape, optimizer=optimizer_cls(learning_rate))

    train_idx, test_idx = utils.train_test_split(
        X, Y, test_size=test_size, use_examples_num=use_examples_num,
        random_state=random_state)

    # global_step = restore_train_state(net)
    global_step = 0

    X_train = X[train_idx]
    X_test = X[test_idx]
    Y_train = Y[train_idx]
    Y_test = Y[test_idx]

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
                net.train_step(X=X_batch, Y=Y_batch)

                reporter.step(global_step)
                global_step += 1

    # save_train_state(net, global_step)
    reporter.finalize()
