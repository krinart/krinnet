import tensorflow as tf

from krinnet import executor as krn_executor


class StorageError(Exception):
    pass


def _save_value(name, value, dtype, executor):
    with executor.context():
        var = tf.get_variable('var_{}'.format(name), dtype=dtype, initializer=value)

        dimensions = tf.get_variable(
            'var_{}_dimensions'.format(name), dtype=tf.int32,
            initializer=len(var.shape))

        executor.initialize(var, dimensions)

        return tf.get_variable(
            'var_{}_shape'.format(name), dtype=tf.int32,
            initializer=var.shape)


def _restore_value(name, dtype, executor):
    with executor.context():
        dimensions = tf.get_variable('var_{}_dimensions'.format(name), shape=(), dtype=tf.int32)
        executor.restore_model()

        shape = tf.get_variable(
            'var_{}_shape'.format(name),
            shape=executor.run(dimensions),
            dtype=tf.int32)
        executor.restore_model()

        var = tf.get_variable('var_{}'.format(name),
                              shape=list(executor.run(shape)), dtype=dtype)
        executor.restore_model()

    return executor.run(var)


def save_values(name, values_dict):
    executor = krn_executor.Executor(name)

    shapes = []
    for name, (value, dtype) in values_dict.items():
        shapes.append(_save_value(name, value, dtype, executor))

    executor.initialize(*shapes)
    executor.save_model()


def restore_values(name, values_dict):
    executor = krn_executor.Executor(name)

    result_dict = {}
    for name, dtype in values_dict.items():
        try:
            result_dict[name] = _restore_value(name, dtype, executor)
        except Exception as e:
            raise StorageError from e

    return result_dict
