import logging
import os
import re
from pathlib import Path

import tensorflow as tf


class Dataset(object):
    def __init__(self, config, types_to_load=None, validate_type_to_load=None):
        self.logger = logging.getLogger("main")
        self.config = config
        self.dataset, self.dataset_val = self.load_data(types_to_load, validate_type_to_load)
        self.feed = iter(self.dataset)
        if self.dataset_val is not None:
            self.feed_val = iter(self.dataset_val)
        else:
            self.feed_val = None

    def load_data(self, types_to_load=None, validate_type_to_load=None):
        dataset = self._load_data(types_to_load)
        dataset_val = None
        if validate_type_to_load is not None:
            dataset_val = self._load_data([validate_type_to_load], load_live=False)
        return dataset, dataset_val

    def _load_data(self, types_to_load=None, load_live=True):
        data_path = Path(self.config.args.data_path)
        mode = self.config.args.mode
        self.logger.info("Loading data from data dir {} with types {}".format(data_path.absolute(), types_to_load))
        data_samples = []
        image_types = ('*.png', '*.bmp')
        fake_count = 0
        for image_type in image_types:
            for path in data_path.rglob(image_type):
                path_string = str(path)
                fake = 'live/' not in path_string.lower()
                if mode in path_string.lower():
                    if fake:
                        if types_to_load is not None:
                            type_index = -3 if 'LivDet2009' in path_string else -2
                            type = re.sub(r'\s+|\d+|_|-', '', path.parts[type_index]).lower()
                            if type not in types_to_load:
                                continue
                        fake_count += 1
                    if fake or load_live:
                        data_samples.append(str(path.absolute()))

        data_sample_count = len(data_samples)
        dataset = self.prepare_for_training(data_samples, data_sample_count, mode)

        self.logger.info("Loaded {} data samples, {} fake and {} live"
                         .format(data_sample_count, fake_count, data_sample_count - fake_count))

        return dataset

    def _process_path(self, file_path):
        def get_label(file_path):
            # The path contains live if it is a live sample
            fake_bool = tf.math.logical_not(tf.strings.regex_full_match(file_path, ".*(?i)live/.*"))
            fake_float = tf.dtypes.cast(fake_bool, tf.float32)
            return tf.reshape(fake_float, [1])

        def decode_img(parts, img):
            # convert the compressed string to a 3D uint8 tensor
            png_bool = tf.strings.regex_full_match(parts[-1], ".*(?i)png.*")
            if png_bool:
                img = tf.image.decode_png(img, channels=3)
            else:
                img = tf.image.decode_bmp(img)
                img = tf.cond(tf.shape(img)[2] == 1, lambda: tf.image.grayscale_to_rgb(img), lambda: img)
            # Use `convert_image_dtype` to convert to floats in the [0,1] range.
            img = tf.image.convert_image_dtype(img, tf.float32)
            # resize the image to the desired size.
            img_size = self.config.IMG_SIZE
            return tf.image.resize(img, [img_size, img_size])

        def get_spoof_type(label, file_path, parts):
            return tf.cond(tf.equal(label, 1.),
                           lambda: get_spoof_type_spoof(file_path, parts),
                           lambda: tf.constant('live'))

        def get_spoof_type_spoof(file_path, parts):
            spoof_index = tf.cond(tf.strings.regex_full_match(file_path, ".*LivDet2009.*"),
                                  lambda: tf.constant(-3),
                                  lambda: tf.constant(-2))
            return tf.strings.regex_replace(tf.strings.lower(parts[spoof_index]), r'\s+|\d+|_|-', '')

        def get_sensor_type(parts):
            return tf.strings.regex_replace(parts[6], r'(?i)(test|train|_)', '')

        parts = tf.strings.split(file_path, os.path.sep)

        label = get_label(file_path)
        spoof_type = get_spoof_type(label, file_path, parts)
        sensor_type = get_sensor_type(parts)
        dataset_name = parts[4]
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(parts, img)
        tf.ensure_shape(label, [1])
        tf.ensure_shape(img, [self.config.IMG_SIZE, self.config.IMG_SIZE, 3])
        return img, label, spoof_type, sensor_type, dataset_name

    def prepare_for_training(self, data_samples, data_sample_count, mode):
        dataset = tf.data.Dataset.from_tensor_slices(data_samples)
        dataset = dataset.shuffle(buffer_size=data_sample_count + 1)
        dataset = dataset.interleave(
            lambda x: tf.data.Dataset.from_tensors(x).map(self._process_path, num_parallel_calls=1),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            deterministic=False
        )
        dataset = dataset.cache()

        # Repeat forever if training
        if mode == 'train':
            dataset = dataset.repeat()

        dataset = dataset.batch(self.config.args.batch_size, drop_remainder=True)

        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset
