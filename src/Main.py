import logging
import sys
import traceback
from argparse import ArgumentParser

from Config import Config
from Dataset import Dataset
from Trainer import Trainer


def main(args=None):
    try:
        config = Config(args)

        dataset = Dataset(config)

        trainer = Trainer(config)
        trainer.train(dataset, None)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        logging.fatal(e)
        logging.debug(traceback.format_exc())
        sys.exit(sys.exc_info()[1])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20, help="Epochs")
    parser.add_argument("--steps", type=int, default=180, help="Steps per epoch")
    parser.add_argument("--steps_val", type=int, default=180, help="Steps per epoch validation")
    parser.add_argument("--max_epoch", type=int, default=40, help="Max epoch")

    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--height", type=int, default=4, help="Height of the DTN")
    parser.add_argument("--data_path", default="../data", help="Path for data folder")
    parser.add_argument("--logging_path", default="../logs", help="Path for logging folder")
    parser.add_argument("--logging", default="DEBUG",
                        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"], help="Logging level")
    parser.add_argument("--plot", type=bool, default=False, help="Plot the training to files in the logs folder")
    parser.add_argument("--mode", default="train",
                        choices=["train", "test"], help="Train or test")

    main(parser.parse_args())
