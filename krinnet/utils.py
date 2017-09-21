from functools import reduce
import operator

import numpy as np
import tensorflow as tf
from sklearn import model_selection


def ensure_tensor_dimensionality(tensor, n_dimensions):
    current_n_dimensions = len(tensor.shape)

    if current_n_dimensions == n_dimensions:
        return tensor

    if n_dimensions == current_n_dimensions + 1:
        return tf.expand_dims(tensor, -1)

    if n_dimensions == 2:
        new_dim = reduce(operator.mul, filter(None, [d.value for d in tensor.shape]), 1)
        return tf.reshape(tensor, (-1, new_dim))

    raise ValueError(
        'Not supported: tensor.shape={}, n_dimensions={}'.format(tensor.shape, n_dimensions))


def cast_shape_to_dimensionality(shape, n_dimensions):
    shape = list(shape)

    current_n_dimensions = len(shape)

    if current_n_dimensions == n_dimensions:
        return shape

    if n_dimensions == current_n_dimensions + 1:
        return shape + [1]

    if n_dimensions == 2:
        new_dim = reduce(operator.mul, filter(None, shape), 1)
        return [-1, new_dim]

    raise ValueError('Not supported: shape={}, n_dimensions={}'.format(shape, n_dimensions))


def train_test_split(X, Y, test_size=.2, use_examples_num=None, random_state=None):
    # First - use only required number of examples
    if use_examples_num:
        if use_examples_num > X.shape[0]:
            raise ValueError('Too big total_size')

        cv = model_selection.StratifiedShuffleSplit(
            n_splits=1, test_size=use_examples_num, random_state=random_state)
        _, index = next(cv.split(X, np.argmax(Y, axis=1)))
        X, Y = X[index], Y[index]

    if not test_size:
        return X, np.array([]), Y, np.array([])

    # Second - split train/test
    cv = model_selection.StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(cv.split(X, np.argmax(Y, axis=1)))
    X_train, X_test, Y_train, Y_test = X[train_idx], X[test_idx], Y[train_idx], Y[test_idx]

    return X_train, X_test, Y_train, Y_test


def batch_iterator(X, Y, batch_size, random_state=None):
    num_examples = X.shape[0]

    n_splits = (num_examples // batch_size) + bool(num_examples % batch_size)

    cv = model_selection.StratifiedKFold(n_splits=n_splits, random_state=random_state)

    train_batch_iterator = (
        (X[index], Y[index])
        for _, index in cv.split(X, np.argmax(Y, axis=1)))

    return train_batch_iterator


def measure_accuracy(Y_pred, Y):
    assert Y_pred.shape == Y.shape
    return np.equal(np.argmax(Y_pred, axis=1), np.argmax(Y, axis=1)).mean()


def reset():
    try:
        tf.get_default_session().close()
    except:
        pass
    tf.reset_default_graph()


def initialize_variable(initializer, default_initializer, random_state=None):
    if initializer is None:
        initializer = default_initializer

    if isinstance(initializer, (int, float, np.ndarray)):
        initializer = tf.constant_initializer(initializer)
    elif callable(initializer):
        try:
            initializer = initializer(seed=random_state)
        except TypeError:
            pass

    return initializer


def linear_image_transform(source, target, steps=10):
    diff = target - source
    step_diff = diff / steps

    images = []
    for step in range(steps+1):
        images.append(source + step_diff*step)

    return images
