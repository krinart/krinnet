import contextlib
import os
import shutil

import tensorflow as tf


class Executor(object):
    def __init__(self, summary_logdir=None, clean_logdir=False):
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)

        self.context = self.graph.as_default()

        self.summary_writer = None

        if summary_logdir:
            if os.path.exists(summary_logdir):
                if not clean_logdir:
                    raise RuntimeError('Specified logdir {} already exists'.format(summary_logdir))
                shutil.rmtree(summary_logdir)

            self.summary_writer = tf.summary.FileWriter(summary_logdir)

    @contextlib.contextmanager
    def initialize(self):
        with self.context:
            yield

            self.session.run(tf.global_variables_initializer())
            self.write_graph(self.graph)

    def run(self, fetches, feed_dict=None):
        return self.session.run(fetches, feed_dict=feed_dict)

    def write_summary(self, step, summary_value):
        if self.summary_writer and summary_value:
            self.summary_writer.add_summary(summary_value, step)

    def write_graph(self, graph):
        if self.summary_writer:
            self.summary_writer.add_graph(graph)

    def finalize(self):
        if self.summary_writer:
            self.summary_writer.flush()

