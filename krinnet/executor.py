import tensorflow as tf

from krinnet import utils


class Executor(object):
    def __init__(self):
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)

        self.context = self.graph.as_default()

    def initialize(self):
        with self.graph.as_default():
            self.session.run(tf.global_variables_initializer())

    def run(self, fetches, feed_dict=None):
        return self.session.run(fetches, feed_dict=feed_dict)

    def save_model(self, path, model_name, force=False):
        if not force:
            path = utils.verify_path_is_empty(path)

        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.save(self.session, '{}/{}.ckpt'.format(path, model_name))

        return path

    def restore_model(self, path, model_name):
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self.session, '{}/{}.ckpt'.format(path, model_name))

        return path
