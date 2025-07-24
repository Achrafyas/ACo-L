import argparse

def args_parser():
    """
    Parse command-line arguments for federated learning experiments.

    Returns:
        args: Namespace with all the parsed arguments.
    """
    parser = argparse.ArgumentParser()

    # -------------------- Federated learning settings --------------------
    parser.add_argument(
        '--epochs',
        type=int,
        default=1,
        help="Number of global federated training rounds."
    )
    parser.add_argument(
        '--num_users',
        type=int,
        default=1,
        help="Total number of client devices (agents)."
    )
    parser.add_argument(
        '--frac',
        type=float,
        default=0.1,
        help="Fraction of clients participating per round (C)."
    )
    parser.add_argument(
        '--local_ep',
        type=int,
        default=1,
        help="Number of local training epochs per client (E)."
    )
    parser.add_argument(
        '--local_bs',
        type=int,
        default=3,
        help="Local mini-batch size for training on each client (B)."
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        help="Learning rate for local optimizers."
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.5,
        help="Momentum parameter for SGD."
    )

    # -------------------- Model architecture --------------------
    parser.add_argument(
        '--model',
        type=str,
        default='mlp',
        help="Model type: 'mlp' or 'cnn'."
    )
    parser.add_argument(
        '--kernel_num',
        type=int,
        default=9,
        help="Number of convolutional kernels (for CNN models)."
    )
    parser.add_argument(
        '--kernel_sizes',
        type=str,
        default='3,4,5',
        help="Comma-separated list of kernel sizes (for CNN)."
    )
    parser.add_argument(
        '--num_channels',
        type=int,
        default=1,
        help="Number of input channels (e.g., 1 for MNIST)."
    )
    parser.add_argument(
        '--norm',
        type=str,
        default='batch_norm',
        help="Normalization layer: 'batch_norm', 'layer_norm', or None."
    )
    parser.add_argument(
        '--num_filters',
        type=int,
        default=32,
        help="Number of filters in conv layers."
    )
    parser.add_argument(
        '--max_pool',
        type=str,
        default='True',
        help="Whether to use max pooling instead of strided conv."
    )

    # -------------------- Dataset & device --------------------
    parser.add_argument(
        '--dataset',
        type=str,
        default='mnist',
        help="Dataset name: 'mnist', 'fmnist', 'cifar', etc."
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=10,
        help="Number of target classes."
    )
    parser.add_argument(
        '--gpu',
        default=None,
        help="GPU ID to use; default is CPU."
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='sgd',
        help="Optimizer type: 'sgd' or 'adam'."
    )

    # -------------------- Data partitioning --------------------
    parser.add_argument(
        '--iid',
        type=int,
        default=1,
        help="1 for IID data split, 0 for non-IID."
    )
    parser.add_argument(
        '--unequal',
        type=int,
        default=0,
        help="1 for unequal non-IID splits, 0 for equal."
    )

    # -------------------- Miscellaneous --------------------
    parser.add_argument(
        '--stopping_rounds',
        type=int,
        default=10,
        help="Rounds of early stopping patience."
    )
    parser.add_argument(
        '--verbose',
        type=int,
        default=1,
        help="Verbosity level (0 silent, higher more verbose)."
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help="Random seed for reproducibility."
    )

    args = parser.parse_args()
    return args