import logging
import os
from pathlib import Path

import tensorflow as tf


class Dataset(object):
    def __init__(self, config):
        self.config = config
        self.autotune = tf.data.experimental.AUTOTUNE
        self.train_dataset = self.load_data()
        self.feed = iter(self.train_dataset)

    def load_data(self):
        data_path = Path(self.config.args.data_path)
        logging.info("Loading data from data dir {}".format(data_path.absolute()))
        data_samples = []
        types = ('*.png', '*.bmp')
        for t in types:
            for path in data_path.rglob(t):
                if self.config.args.mode in str(path).lower():
                    data_samples.append(str(path.absolute()))

        list_dataset = tf.data.Dataset.from_tensor_slices(data_samples)
        labeled_dataset = list_dataset.map(self._process_path, num_parallel_calls=self.autotune)
        train_dataset = self.prepare_for_training(labeled_dataset)

        logging.info("Loaded {} data samples".format(len(data_samples)))

        return train_dataset

    def _process_path(self, file_path):
        def get_label(file_path):
            # convert the path to a list of path components
            parts = tf.strings.split(file_path, os.path.sep)
            # The second to last is the class-directory
            fake_bool = tf.strings.regex_full_match(parts[-3], ".*(?i)(fake|spoof).*")
            fake_float = tf.dtypes.cast(fake_bool, tf.float32)
            return tf.reshape(fake_float, [1])

        def decode_img(file_path, img):
            # convert the compressed string to a 3D uint8 tensor
            parts = tf.strings.split(file_path, os.path.sep)
            png_bool = tf.strings.regex_full_match(parts[-1], ".*(?i)png.*")
            if png_bool:
                img = tf.image.decode_png(img, channels=3)
            else:
                img = tf.image.decode_bmp(img, channels=1)
            # Use `convert_image_dtype` to convert to floats in the [0,1] range.
            img = tf.image.convert_image_dtype(img, tf.float32)
            # resize the image to the desired size.
            img_size = self.config.IMG_SIZE
            return tf.image.resize(img, [img_size, img_size])

        label = get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(file_path, img)
        tf.ensure_shape(label, [1])
        tf.ensure_shape(img, [self.config.IMG_SIZE, self.config.IMG_SIZE, 3])
        return img, label

    def prepare_for_training(self, dataset, cache=True, shuffle_buffer_size=1000):
        # This is a small dataset, only load it once, and keep it in memory.
        # use `.cache(filename)` to cache preprocessing work for datasets that don't
        # fit in memory.
        if cache:
            if isinstance(cache, str):
                dataset = dataset.cache(cache)
            else:
                dataset = dataset.cache()

        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

        # Repeat forever
        dataset = dataset.repeat()

        dataset = dataset.batch(self.config.args.batch_size, drop_remainder=True)

        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        dataset = dataset.prefetch(buffer_size=self.autotune)

        return dataset
