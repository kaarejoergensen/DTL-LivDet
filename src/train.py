from argparse import ArgumentParser


def main(args=None):
    print(args)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--gpu", type=float, default=0, help="GPU to use")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--resblocks", type=int, default=19, help="Number of resblocks")
    parser.add_argument("--print_loss", type=bool, default=False, help="Print loss at each epoch")
    parser.add_argument("--loss_function", default="MSE",
                        choices=["MSE", "MAE"], help="Choose loss function")
    parser.add_argument("--features", type=int, default=128, help="Number of features in each layer")

    main(parser.parse_args())
