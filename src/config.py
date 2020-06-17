import tensorflow as tf
import logging
import os


class Config(object):
    IMG_SIZE = 224

    def __init__(self, args=None):
        self.args = args
        if not os.path.exists(args.log):
            os.mkdir(args.log)
        logging.basicConfig(level=logging.getLevelName(args.logging),
                            format="%(asctime)s [%(levelname)s] %(message)s",
                            handlers=[
                                logging.FileHandler("{}/main.log".format(args.log)),
                                logging.StreamHandler()
                            ])
        logging.info("Arguments: {}".format(args))
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_memory_growth(gpus, True)
                tf.config.experimental.set_visible_devices(gpus, 'GPU')
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                logging.info("Using {} of {} total gpus".format(logical_gpus, gpus))
            except RuntimeError as e:
                logging.warning("Error on GPU initialization:")
                logging.debug(e)
        else:
            logging.info("No GPUs available")