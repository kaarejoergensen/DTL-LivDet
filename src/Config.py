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
        file_path = "{}/main.log".format(logging_path)
        rotating_file_handler = RotatingFileHandler(filename=file_path, backupCount=5)
        if os.path.isfile(file_path):
            rotating_file_handler.doRollover()
        logging.basicConfig(level=logging.getLevelName(args.logging),
                            format="%(asctime)s [%(levelname)s] %(message)s",
                            handlers=[
                                rotating_file_handler,
                                logging.StreamHandler()
                            ])
        logging.info("Arguments: {}".format(args))
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            logging.info("Using gpus: {}".format(gpus))
        else:
            logging.info("No GPUs available")