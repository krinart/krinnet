import tensorflow as tf
from tqdm import tqdm

from krinnet import reporting
from krinnet import utils


def train_nn(net, X, Y, epochs, learning_rate, optimizer_cls=tf.train.AdamOptimizer, batch_size=512,
             random_state=None, summary_step=5, clean_logdir=False,
             test_size=.2, use_examples_num=None, print_accuracy_step=None, print_error_step=None):

    net.build(input_shape=X.shape, optimizer=optimizer_cls(learning_rate))

    reporter = net.name and reporting.Reporter(
        net,
        summary_logdir='logs/{}'.format(net.name),
        clean_logdir=clean_logdir,
        summary_step=summary_step,
        print_accuracy_step=print_accuracy_step,
        print_error_step=print_error_step,
    )

    X_train, X_test, Y_train, Y_test = utils.train_test_split(
        X, Y, test_size=test_size, use_examples_num=use_examples_num,
        random_state=random_state)

    global_step = 0

    for _ in tqdm(range(epochs)):
        batches = utils.batch_iterator(X_train, Y_train, batch_size, random_state=random_state)

        for X_batch, Y_batch in batches:
            net.train_step(X=X_batch, Y=Y_batch)

            if reporter:
                reporter.step(
                    global_step, X_train, X_test, Y_train=Y_train, Y_test=Y_test)

            global_step += 1

    net.executor.finalize()
