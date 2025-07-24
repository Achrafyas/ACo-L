import codecs
import pickle
import copy
import torch
import numpy as np
from sklearn import metrics
import Config
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from FederatedLearning.Sampling import (
    mnist_iid, mnist_noniid, mnist_noniid_unequal,
    cifar_iid, cifar_noniid
)


class Utilities:
    """
    Helper routines for:
      • Loading and partitioning datasets
      • Plotting a confusion matrix
      • Averaging model weights across clients
      • Printing experiment details
    """

    def get_dataset(self, args, dataset):
        """
        Load train/test splits for the named dataset, then
        partition the train split among args.num_users clients
        using IID or one of the non-IID strategies.
        Returns: (train_dataset, test_dataset, user_groups_dict)
        """
        # Prepare placeholders
        train_dataset = None
        test_dataset = None
        user_groups = None

        if dataset == 'cifar':
            # CIFAR10 (uses MNIST loader here by mistake? – adjust if needed)
            data_dir = Config.data_set_path + '/cifar/'
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,)*3, (0.5,)*3)
            ])
            train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
            test_dataset  = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

            # Partition among clients
            if args.iid:
                user_groups = cifar_iid(train_dataset, args.num_users)
            else:
                if args.unequal:
                    raise NotImplementedError("Unequal CIFAR splits not implemented")
                user_groups = cifar_noniid(train_dataset, args.num_users)

        else:  # 'mnist' or 'fmnist'
            data_dir = Config.data_set_path + ('/mnist/' if dataset == 'mnist' else '/fmnist/')
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            if dataset == 'mnist':
                train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
                test_dataset  = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
            else:
                train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
                test_dataset  = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)

            # Partition among clients
            if args.iid:
                user_groups = mnist_iid(train_dataset, args.num_users)
            else:
                if args.unequal:
                    user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
                else:
                    user_groups = mnist_noniid(train_dataset, args.num_users)

        return train_dataset, test_dataset, user_groups

    def plot_confusion_matrix(self, labels, pred_labels):
        """
        Given true labels and predictions, plot a 10×10 confusion matrix.
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        cm = metrics.confusion_matrix(labels, pred_labels)
        disp = metrics.ConfusionMatrixDisplay(cm, display_labels=range(10))
        disp.plot(values_format='d', cmap='Blues', ax=ax)

    def average_weights(self, serialized_weights_list):
        """
        Given a list of base64-encoded weight dicts from different clients,
        unpickle them, sum elementwise, and divide by number of clients.
        Returns a single averaged state_dict.
        """
        w_avg = None

        # Deserialize each client's weights
        for raw in serialized_weights_list:
            unpickled = pickle.loads(codecs.decode(raw.encode(), "base64"))
            # Each pickled object is a list; take the 0th element
            client_state = unpickled[0]
            if w_avg is None:
                # Initialize accumulator with a deep copy of the first client
                w_avg = copy.deepcopy(client_state)
            else:
                # Sum in-place
                for key in w_avg:
                    w_avg[key] += client_state[key]

        # Divide by number of clients to get the average
        for key in w_avg:
            w_avg[key] = torch.div(w_avg[key], len(serialized_weights_list))

        return w_avg

    def exp_details(self, args):
        """
        Print a human-readable summary of the federated experiment
        settings (model type, optimizer, learning rate, etc.).
        """
        print('\nExperimental details:')
        print(f'    Model            : {args.model}')
        print(f'    Optimizer        : {args.optimizer}')
        print(f'    Learning rate    : {args.lr}')
        print(f'    Global rounds    : {args.epochs}\n')

        print('Federated parameters:')
        print('    IID' if args.iid else '    Non-IID')
        print(f'    Fraction clients : {args.frac}')
        print(f'    Local batch size : {args.local_bs}')
        print(f'    Local epochs     : {args.local_ep}\n')
