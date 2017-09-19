import tensorflow as tf
from tqdm import tqdm

from krinnet import executor as krn_executor
from krinnet import reporting
from krinnet import utils


def build(net, X, Y, optimizer=None, summary_logdir=None, summary_step=5, clean_logdir=False):
    reporter = None

    executor = krn_executor.Executor(summary_logdir=summary_logdir, clean_logdir=clean_logdir)

    with executor.initialize():
        net.set_executor(executor)

        net.build(X, Y)

        if summary_logdir:
            reporter = reporting.Reporter(
                executor,
                net,
                summary_logdir=summary_logdir,
                clean_logdir=clean_logdir,
                summary_step=summary_step)

        with tf.variable_scope('optimizer'):
            minimizer = optimizer and optimizer.minimize(net.get_loss_tensor())

    return executor, minimizer, reporter


def train_nn(net, X, Y, epochs, learning_rate, optimizer_cls=tf.train.AdamOptimizer, batch_size=512,
             random_state=None, summary_logdir=None, summary_step=5, clean_logdir=False,
             test_size=.2, use_examples_num=None, print_accuracy_step=None, print_error_step=None):

    optimizer = optimizer_cls(learning_rate)

    executor, minimizer, reporter = build(net, X, Y, optimizer, summary_logdir=summary_logdir,
                                          summary_step=summary_step, clean_logdir=clean_logdir)

    X_train, X_test, Y_train, Y_test = utils.train_test_split(
        X, Y, test_size=test_size, use_examples_num=use_examples_num,
        random_state=random_state)

    global_step = 0

    for _ in tqdm(range(epochs)):
        batches = utils.batch_iterator(X_train, Y_train, batch_size, random_state=random_state)

        for X_batch, Y_batch in batches:
            net.train_step(minimizer, X=X_batch, Y=Y_batch)

            if reporter:
                reporter.step(
                    global_step, X_train, X_test, Y_train=Y_train, Y_test=Y_test,
                    print_accuracy_step=print_accuracy_step, print_error_step=print_error_step)

            global_step += 1

    executor.finalize()
