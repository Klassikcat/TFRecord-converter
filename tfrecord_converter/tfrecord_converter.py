import tensorflow as tf
import os
import random
from tqdm.notebook import tqdm
import argparse
import numpy as np
import math
import pandas as pd
from tfrecord_converter.config import SETTINGS

settings = SETTINGS()

class Converter:
    def __init__(self, tfrecord_writer, example_rule,
                 origin_path, df, output_dir,
                 n_shards, SEED=int):
        """
        :param tfrecord_converter:
        :param origin_path:
        :param df:
        :param output_dir:
        :param n_shards:
        :param SEED:
        """
        self.tfrecord_writer = tfrecord_writer
        self.example_rule = example_rule
        self.origin_path = origin_path
        self.df = df
        self.output_dir = output_dir
        self.n_shards = n_shards
        self.SEED = SEED

    def iterator(self, dataframe, index, target):
        lists = dataframe[target].iloc[index]
        random.Random(self.SEED).shuffle(lists)
        return lists

    def _get_shard_path(self, shard_id, shard_size):
        return os.path.join(self.origin_path,
                            self.output_dir,
                            f'{shard_id:03d}-{shard_size}.tfrecord')

    def write_tfrecord_file(self, df, indices, shard_path):
        options = tf.io.TFRecordOptions(compression_type='GZIP')
        with tf.io.TFRecordWriter(shard_path, options=options) as out:
            for index in tqdm(indices):
                example = self.tfrecord_writer(df, index, self.example_rule)
                out.write(example.SerializeToString())

    def convert(self):
        size = len(self.df)
        offset = 0
        shard_size = math.ceil(size/self.n_shards)
        cumulative_size = offset + size
        for shard_id in range(1, self.n_shards + 1):
            step_size = min(shard_size, cumulative_size - offset)
            shard_path = self._get_shard_path(shard_id, step_size)
            file_indices = np.arange(offset, offset + step_size)
            self.write_tfrecord_file(self.df, file_indices, shard_path)
            offset += step_size

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--origin-path', type=str, dest='origin_path',
                        help='Absolute path of the target path.'
                             '(example = /User/administrator/project-path/contents/)')
    parser.add_argument('--data-path', type=str, dest='data_path',
                        default=settings._DEFAULT_DATA_PATH,
                        help='relative path of the data path in the project file.'
                             f'(default: {settings._DEFAULT_DATA_PATH})')
    parser.add_argument('--label-path', type=str, dest='label_path',
                        default=settings._DEFAULT_LABEL_PATH,
                        help='relative path of the label path in the project file.'
                             f'(defaults: {settings._DEFAULT_LABEL_PATH})')
    parser.add_argument('--output-dir', type=str, dest='output_dir',
                        default=settings._DEFAULT_OUTPUT_DIR,
                        help='relative directory in the project'
                             'that tfrecord file will be saved.'
                             f'(defaults:{settings._DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--n-shards', type=int, dest='n_shards', default=1,
                        help='number of shards to divide dataset TFRecord into.'
                             '(defaults: 1)')
    return parser.parse_args()

def main(args):
    converter = Converter(args.origin_path,
                          args.data_path,
                          args.label_path,
                          args.output_dir,
                          args.n_shards)
    converter.convert()

if __name__ == '__main__':
    main(parse_args())
