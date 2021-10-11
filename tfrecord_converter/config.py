import tensorflow as tf


class SETTINGS:
    _DEFAULT_DATA_PATH = 'data'
    _DEFAULT_LABEL_PATH = 'label'
    _DEFAULT_OUTPUT_DIR = ''

    _DEFAULT_N_SHARDS = 1

    batch_size = 32
    dropout_rate = 0.2


def _float_feature(list_of_floats):  # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))





