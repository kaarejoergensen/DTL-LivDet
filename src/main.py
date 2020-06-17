from argparse import ArgumentParser

from config import Config
from dataset import Dataset


def main(args=None):
    config = Config(args)
    dataset = Dataset(config)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20, help="Epochs")
    parser.add_argument("--steps", type=int, default=180, help="Steps per epoch")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--data", default="../data", help="Data dir")
    parser.add_argument("--log", default="../logs", help="Log dir")
    parser.add_argument("--logging", default="DEBUG",
                        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"], help="Logging level")
    parser.add_argument("--mode", default="train",
                        choices=["train", "test"], help="Train or test")

    main(parser.parse_args())
