import logging
import os
from logging.handlers import RotatingFileHandler

import tensorflow as tf


class Config(object):
    IMG_SIZE = 256
    TRU_PARAMETERS = {
        "alpha": 1e-3,
        "beta": 1e-2,
        "mu_update_rate": 1e-3,
    }

    def __init__(self, args=None):
        self.args = args
        logging_path = args.logging_path
        if not os.path.exists(logging_path):
            os.mkdir(logging_path)
        main_logger = setup_main_logger('main', logging_path, logging.getLevelName(args.logging))
        if args.mode == 'test':
            setup_test_logger('test_spoof', logging_path, logging.getLevelName(args.logging))
            setup_test_logger('test_live', logging_path, logging.getLevelName(args.logging))
        main_logger.info("Arguments: {}".format(args))
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            main_logger.info("Using gpus: {}".format(gpus))
        else:
            main_logger.info("No GPUs available")


def setup_main_logger(name, logging_path, level):
    file_path = "{}/{}.log".format(logging_path, name)
    handlers = [
        create_rotating_file_handler(file_path),
        logging.StreamHandler()
    ]
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    return setup_logger(name, handlers, formatter, level)


def setup_test_logger(name, logging_path, level):
    file_path = "{}/{}.log".format(logging_path, name)
    handlers = [
        create_rotating_file_handler(file_path)
    ]
    formatter = logging.Formatter("%(message)s")
    return setup_logger(name, handlers, formatter, level)


def create_rotating_file_handler(file_path, backup_count=10):
    rotating_file_handler = RotatingFileHandler(filename=file_path, backupCount=backup_count)
    if os.path.isfile(file_path):
        rotating_file_handler.doRollover()
    return rotating_file_handler


def setup_logger(name, handlers, formatter, level):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
