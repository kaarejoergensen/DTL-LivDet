import logging
import os

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
        logging.basicConfig(level=logging.getLevelName(args.logging),
                            format="%(asctime)s [%(levelname)s] %(message)s",
                            handlers=[
                                logging.FileHandler("{}/main.log".format(logging_path)),
                                logging.StreamHandler()
                            ])
        logging.info("Arguments: {}".format(args))
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            logging.info("Using gpus: {}".format(gpus))
            # try:
            #     tf.config.experimental.set_memory_growth(gpus, True)
            #     tf.config.experimental.set_visible_devices(gpus, 'GPU')
            #     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            #     logging.info("Using {} of {} total gpus".format(logical_gpus, gpus))
            # except RuntimeError as e:
            #     logging.warning("Error on GPU initialization:")
            #     logging.debug(e)
        else:
            logging.info("No GPUs available")