import tensorflow as tf


class Executor(object):
    def __init__(self):
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)

        self.context = self.graph.as_default()
        self.summary_writer = None

    def initialize(self):
        with self.graph.as_default():
            self.session.run(tf.global_variables_initializer())
            self.write_graph(self.graph)

    def run(self, fetches, feed_dict=None):
        return self.session.run(fetches, feed_dict=feed_dict)

    def write_graph(self, graph):
        if self.summary_writer:
            self.summary_writer.add_graph(graph)

    def finalize(self):
        if self.summary_writer:
            self.summary_writer.flush()
