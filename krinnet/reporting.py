import tensorflow as tf

from krinnet import context


class Reporter(object):
    def __init__(self, executor, net, summary_logdir, clean_logdir=False, summary_step=5):
        self.executor = executor
        self.net = net
        self.summary_logdir = summary_logdir
        self.clean_logdir = clean_logdir
        self.summary_step = summary_step

        self.train_summaries = self.net.train_summaries and tf.summary.merge(
            self.net.train_summaries)

        self.test_summaries = self.net.test_summaries and tf.summary.merge(self.net.test_summaries)
        self.stat_summaries = self.net.stat_summaries and tf.summary.merge(self.net.stat_summaries)

    def write_summary(self, step, summary, feed_dict=None, return_accuracy=False,
                      return_error=False):
        evaluate_tensors = []
        if return_accuracy:
            evaluate_tensors.append('accuracy_layer/accuracy:0')
        if return_error:
            evaluate_tensors.append('loss_layer/loss:0')

        summary_value, *return_value = self.executor.run(
            [summary] + evaluate_tensors, feed_dict=feed_dict)

        self.executor.write_summary(step, summary_value)

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
            feed_dict=self.net.get_train_feed(context=run_context),
            return_accuracy=return_accuracy,
            return_error=return_error)

    def write_stat_summary(self, step):
        if self.stat_summaries is None:
            return

        return self.write_summary(step, self.stat_summaries)

    def step(self, step, X_train, X_test, Y_train=None, Y_test=None, print_accuracy_step=None,
             print_error_step=None):
        if print_accuracy_step is not None and print_accuracy_step < self.summary_step:
            raise ValueError('print_accuracy_step ({}) can not be < step ({})'.format(
                print_accuracy_step, self.summary_step))

        if print_error_step is not None and print_error_step < self.summary_step:
            raise ValueError('print_error_step ({}) can not be < step ({})'.format(
                print_error_step, self.summary_step))

        if step % self.summary_step != 0:
            return

        print_accuracies = print_accuracy_step and (step % print_accuracy_step == 0)
        print_error = print_error_step and (step % print_error_step == 0)

        train_accuracy, train_error = self.write_train_summary(
            step, X=X_train, Y=Y_train, return_accuracy=print_accuracies, return_error=print_error)
        test_accuracy, test_error = self.write_test_summary(
            step, X=X_test, Y=Y_test, return_accuracy=print_accuracies, return_error=print_error)
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
