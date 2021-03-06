import contextlib

import tensorflow as tf

from krinnet import utils


class Executor(object):
    def __init__(self, model_name):
        self.model_name = model_name
        self.path = 'models/{}/model.ckpt'.format(model_name)
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)

    @contextlib.contextmanager
    def context(self):
        with self.graph.as_default():
            with self.session.as_default():
                yield

    def initialize(self, *vars):
        with self.context():
            if vars:
                self.run(tf.variables_initializer(vars))
            else:
                self.run(tf.global_variables_initializer())

    def run(self, fetches, feed_dict=None):
        return self.session.run(fetches, feed_dict=feed_dict)

    def save_model(self, force=False):
        # if not force:
        #     path = utils.verify_path_is_empty(self.path)

        with self.context():
            saver = tf.train.Saver()
            saver.save(self.session, self.path)

        return self.path

    def restore_model(self):
        with self.context():
            saver = tf.train.Saver()
            saver.restore(self.session, self.path)

        return self.path
