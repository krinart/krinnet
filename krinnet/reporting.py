import tensorflow as tf

from krinnet import context
from krinnet import utils


class Reporter(object):
    def __init__(self, net, X_train, X_test, summary_logdir, Y_train=None, Y_test=None,
                 summary_step=5, print_accuracy_step=None, print_error_step=None):
        self.net = net
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test

        if print_accuracy_step is not None and print_accuracy_step < summary_step:
            raise ValueError('print_accuracy_step ({}) can not be < step ({})'.format(
                print_accuracy_step, summary_step))
        if print_error_step is not None and print_error_step < summary_step:
            raise ValueError('print_error_step ({}) can not be < step ({})'.format(
                print_error_step, summary_step))
        self.summary_step = summary_step
        self.print_accuracy_step = print_accuracy_step
        self.print_error_step = print_error_step

        self.train_summaries = self.net.train_summaries and tf.summary.merge(
            self.net.train_summaries)

        self.test_summaries = self.net.test_summaries and tf.summary.merge(self.net.test_summaries)
        self.stat_summaries = self.net.stat_summaries and tf.summary.merge(self.net.stat_summaries)

        summary_logdir = utils.verify_path_is_empty(summary_logdir)
        self.summary_writer = tf.summary.FileWriter(summary_logdir)
        self.summary_writer.add_graph(self.net.executor.graph)

    def write_summary(self, step, summary, feed_dict=None, return_accuracy=False,
                      return_error=False):
        evaluate_tensors = []
        if return_accuracy:
            evaluate_tensors.append('accuracy_layer/accuracy:0')
        if return_error:
            evaluate_tensors.append('loss_layer/loss:0')

        summary_value, *return_value = self.net.executor.run(
            [summary] + evaluate_tensors, feed_dict=feed_dict)

        if summary_value:
            self.summary_writer.add_summary(summary_value, step)

        accuracy, error = None, None

        return_value = list(return_value)
        if return_accuracy:
            accuracy = return_value.pop(0)
        if return_error:
            error = return_value.pop(0)

        return accuracy, error

    def write_train_summary(self, step, X, Y=None, return_accuracy=False, return_error=False):
        if self.train_summaries is None or not X.size:
            return None, None

        run_context = context.Context(X=X, Y=Y)
        return self.write_summary(
            step,
            self.train_summaries,
            feed_dict=self.net.get_train_feed(context=run_context),
            return_accuracy=return_accuracy,
            return_error=return_error)

    def write_test_summary(self, step, X, Y=None, return_accuracy=False, return_error=False):
        if self.test_summaries is None or not X.size:
            return None, None

        run_context = context.Context(X=X, Y=Y)
        return self.write_summary(
            step,
            self.test_summaries,
            feed_dict=self.net.get_test_feed(context=run_context),
            return_accuracy=return_accuracy,
            return_error=return_error)

    def write_stat_summary(self, step):
        if self.stat_summaries is None:
            return

        return self.write_summary(step, self.stat_summaries)

    def step(self, step):
        if step % self.summary_step != 0:
            return

        print_accuracies = self.print_accuracy_step and (step % self.print_accuracy_step == 0)
        print_error = self.print_error_step and (step % self.print_error_step == 0)

        train_accuracy, train_error = self.write_train_summary(
            step, X=self.X_train, Y=self.Y_train, return_accuracy=print_accuracies,
            return_error=print_error)

        test_accuracy, test_error = self.write_test_summary(
            step, X=self.X_test, Y=self.Y_test, return_accuracy=print_accuracies,
            return_error=print_error)

        self.write_stat_summary(step)

        if print_accuracies and (train_accuracy is not None or test_accuracy is not None):
            res = ['step={}'.format(step)]
            if train_accuracy is not None:
                res.append('train_accuracy={:.2f}'.format(train_accuracy))
            if test_accuracy is not None:
                res.append('test_accuracy={:.2f}'.format(test_accuracy))
            print(', '.join(res))

        if print_error and (train_error is not None or test_error is not None):
            res = ['step={}'.format(step)]
            if train_error is not None:
                res.append('train_error={:.5f}'.format(train_error))
            if test_error is not None:
                res.append('test_error={:.5f}'.format(test_error))
            print(', '.join(res))

    def finalize(self):
        self.summary_writer.flush()