import copy
import os
import time

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from termcolor import colored

from FederatedLearning.Utilities import Utilities
from FederatedLearning.Models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, CNNCustom
from FederatedLearning.Options import args_parser
from FederatedLearning.Update import LocalUpdate


class Federated:
    """
    Encapsulates all local training logic for one simulated FL participant.
    """

    def __init__(self, agent_name, model_path, dataset, model_type):
        # Basic identifiers and paths
        self.agent_name = agent_name
        self.model_path = model_path
        self.dataset = dataset
        self.model_type = model_type

        # Placeholders for model and training statistics
        self.model = None
        self.global_weights = None
        self.train_accuracy, self.train_loss = [], []
        self.test_accuracy,  self.test_loss  = [], []

        # Timers and logging
        self.start_time = time.time()
        self.logger = SummaryWriter('../logs')  # TensorBoard writer

        # Parse CLI arguments for federated settings
        self.args = args_parser()
        print(colored('=' * 30, 'green'))
        print(self.args)
        print(colored('=' * 30, 'green'))

        # Helpers for data splitting etc.
        self.utilities = Utilities()

        # Choose device (GPU if specified)
        if self.args.gpu:
            torch.cuda.set_device(self.args.gpu)
        self.device = 'cuda' if self.args.gpu else 'cpu'

        # Load dataset and split among "users" for IID or non-IID
        print(self.dataset)
        self.train_dataset, self.test_dataset, self.user_groups = \
            self.utilities.get_dataset(self.args, self.dataset)

    def build_model(self):
        """
        Instantiate the local PyTorch model based on model_type and dataset.
        If a pretrained file is provided, load its weights.
        """
        if self.model_type == 'cnn':
            if self.dataset == 'mnist':
                self.model = CNNMnist(args=self.args)
            elif self.dataset == 'fmnist':
                self.model = CNNFashion_Mnist(args=self.args)
            elif self.dataset == 'cifar':
                self.model = CNNCifar(args=self.args)

        elif self.model_type == 'mlp':
            # Flatten input size
            img_size = self.train_dataset[0][0].shape
            in_features = 1
            for dim in img_size:
                in_features *= dim
            print(img_size)
            self.model = MLP(dim_in=in_features, dim_hidden=32, dim_out=self.args.num_classes)

        elif self.dataset == 'custom':
            self.model = CNNCustom(args=self.args)

        else:
            exit('Error: unrecognized model type')

        # If user supplied a model file, load those weights
        if self.model_path:
            print("Using provided model file:", self.model_path)
            self.model.load_state_dict(torch.load(self.model_path))
            # Print one parameter value as a sanity check
            print(self.model.state_dict()['layer_input.weight'].numpy().flatten()[0])

    def set_model(self):
        """
        Move the model to the correct device and snapshot its state dict.
        """
        self.model.to(self.device)
        self.global_weights = self.model.state_dict()

    def print_model(self):
        """Print model architecture to console."""
        print(self.model)
    
    def get_model(self):
        """
        Return the PyTorch model instance.
        """
        return self.model

    async def train_local_model(self, epoch=1):
        """
        Perform one round of local training:
        - Clone the current model
        - Run LocalUpdate with given local epochs and batch size
        - Save the updated model
        - Compute training and test accuracy/loss
        Returns updated weights, losses, and metrics.
        """
        self.start_time = time.monotonic()

        # Wrap dataset split for this "user"
        local_update = LocalUpdate(
            args=self.args,
            dataset=self.train_dataset,
            idxs=self.user_groups[0],
            logger=self.logger
        )

        # Run asynchronous local update
        self.model, weights, loss = await local_update.update_weights(
            model=copy.deepcopy(self.model),
            global_round=epoch
        )

        # Persist model checkpoint
        torch.save(self.model.state_dict(), "Saved Models/model.pt")
        print("Saved updated model")

        # Collect results
        local_weights = [copy.deepcopy(weights)]
        local_losses  = [copy.deepcopy(loss)]
        print("Local weights snapshot:", local_weights[0])

        # Evaluate on train and test sets
        train_acc, train_loss, test_acc, test_loss = self.get_accuracy(local_update)
        self.end_time = time.monotonic()

        return local_weights, local_losses, train_acc, train_loss, test_acc, test_loss

    def get_accuracy(self, local_update):
        """
        After training, compute and print:
        - Training accuracy & loss (on local data)
        - Test accuracy & loss (on global test set)
        """
        self.model.eval()
        acc, loss = local_update.inference(model=self.model)
        print(f"[{self.agent_name}] Train Accuracy: {acc*100:.2f}%  Loss: {loss:.4f}")
        self.train_accuracy.append(acc)
        self.train_loss.append(loss)

        test_acc, test_loss = local_update.test_inference(self.args, self.model, self.test_dataset)
        print(f"[{self.agent_name}] Test  Accuracy: {test_acc*100:.2f}%  Loss: {test_loss:.4f}")
        self.test_accuracy.append(test_acc)
        self.test_loss.append(test_loss)

        return [
            round(acc*100, 2),
            round(loss,    4),
            round(test_acc*100, 2),
            round(test_loss, 4),
        ]

    def average_all_weights(self, local_weights, local_losses, verbose=False):
        """
        Compute global average of all client updates (classical FedAvg).
        """
        global_w = self.utilities.average_weights(local_weights)
        self.model.load_state_dict(global_w)
        avg_loss = sum(local_losses) / len(local_losses)
        self.train_loss.append(avg_loss)

    def add_new_local_weight_local_losses(self, local_weights, local_losses):
        """
        Directly replace model with new local weights (used in consensus).
        """
        self.model.load_state_dict(local_weights)
        self.train_loss.append(local_losses)

    def get_predictions(self, model, iterator, device):
        """
        Utility to get raw images, labels, and predicted probabilities
        for visualization or analysis.
        """
        model.eval()
        images, labels, probs = [], [], []
        with torch.no_grad():
            for x, y in iterator:
                x = x.to(device)
                logits = model(x)
                prob  = F.softmax(logits, dim=1)
                images.append(x.cpu())
                labels.append(y.cpu())
                probs.append(prob.cpu())

        return torch.cat(images), torch.cat(labels), torch.cat(probs)