import logging
import sys
import traceback
from argparse import ArgumentParser

from Config import Config
from runners.Tester import Tester
from runners.Trainer import Trainer


def main(args=None):
    try:
        config = Config(args)

        if args.mode == 'train':
            trainer = Trainer(config)
            trainer.train()
        else:
            tester = Tester(config)
            tester.test()

    except KeyboardInterrupt:
        raise
    except Exception as e:
        logging.fatal(e)
        logging.debug(traceback.format_exc())
        sys.exit(sys.exc_info()[1])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40, help="Epochs")
    parser.add_argument("--steps", type=int, default=180, help="Steps per epoch")
    parser.add_argument("--steps_val", type=int, default=60, help="Steps per epoch for leave-one-out validation")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--height", type=int, default=4, help="Height of the DTN")
    parser.add_argument("--data_path", default="../data", help="Path for data folder")
    parser.add_argument("--logging_path", default="../logs", help="Path for logging folder")
    parser.add_argument("--checkpoint_path", default="../model", help="Path for checkpoint folder")
    parser.add_argument("--logging", default="DEBUG",
                        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"], help="Logging level")
    parser.add_argument("--plot", action='store_true', help="Plot the training to files in the logs folder")
    parser.add_argument("--ignore_checkpoint", action='store_true', help="Ignore checkpoint and start from scratch")
    parser.add_argument("--keep_running", action='store_true', help="Keep running the training")
    parser.add_argument("--mode", default="train",
                        choices=["train", "test"], help="Train or test")
    common_types = ["ecoflex", "bodydouble", "gelatin", "gelatine", "latex",
                    "modasil", "playdoh", "silgum", "silicone", "woodglue"]
    parser.add_argument("--training_types", default=common_types,
                        help="Specify the different types of fake samples for training (for leave-one-out validation)")
    # parser.add_argument("--testing_types", default=[*common_types, "liquidecoflex", "oomoo", "rtv"],
    #                     help="Specify the different types of fake samples for testing")
    parser.add_argument("--dont_validate", action='store_true', help="Dont use leave-one-out validation")
    parser.add_argument("--log_less", action='store_true',
                        help="Log 10 times each epoch when training instead of each step")

    main(parser.parse_args())
