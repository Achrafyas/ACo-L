import asyncio
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

class DatasetSplit(Dataset):
    """Wraps a full PyTorch Dataset so that only a given subset of indices is visible."""

    def __init__(self, dataset, idxs):
        # Store original dataset and the list of indices this client should see
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        # Length is just how many indices we have
        return len(self.idxs)

    def __getitem__(self, item):
        # Return the image/label pair at the requested index in our subset
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate:
    """
    Manages one client’s local training:
    - Splits its data into train/val/test loaders
    - Runs E epochs of local SGD or Adam
    - Returns updated model weights and loss
    """

    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger

        # Create train, validation, and test DataLoaders for this client
        self.train_loader, self.valid_loader, self.test_loader = self.train_val_test(
            dataset, list(idxs)
        )

        # Device selection (GPU or CPU)
        self.device = 'cuda' if args.gpu else 'cpu'
        # Use negative-log-likelihood loss by default
        self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Split the client’s indices into:
        - 80% train
        - 10% validation
        - 10% test
        and wrap each in a DataLoader.
        """
        n = len(idxs)
        train_end = int(0.8 * n)
        val_end   = int(0.9 * n)

        train_loader = DataLoader(
            DatasetSplit(dataset, idxs[:train_end]),
            batch_size=self.args.local_bs,
            shuffle=True
        )
        valid_loader = DataLoader(
            DatasetSplit(dataset, idxs[train_end:val_end]),
            batch_size=max(1, len(idxs[train_end:val_end]) // 10),
            shuffle=False
        )
        test_loader = DataLoader(
            DatasetSplit(dataset, idxs[val_end:]),
            batch_size=max(1, len(idxs[val_end:]) // 10),
            shuffle=False
        )
        return train_loader, valid_loader, test_loader

    async def update_weights(self, model, global_round, verbose=False):
        """
        Perform local training for self.args.local_ep epochs:
        - Zero gradients, forward, backward, step optimizer
        - Log loss to TensorBoard
        Returns:
          model: the trained model
          state_dict: its final parameters
          avg_loss: average loss over all batches/epochs
        """
        model.train()
        epoch_loss = []

        # Choose optimizer
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.5)
        else:  # 'adam'
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)

        for _ in range(self.args.local_ep):
            batch_losses = []
            for images, labels in self.train_loader:
                # allow other coroutines to run
                await asyncio.sleep(0)

                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                outputs = model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # log to TensorBoard
                self.logger.add_scalar('loss', loss.item())
                batch_losses.append(loss.item())

            epoch_loss.append(sum(batch_losses) / len(batch_losses))

        # Return final model, its parameters, and the average epoch loss
        return model, model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """
        Evaluate the model on this client’s held-out test_loader.
        Returns (accuracy, loss).
        """
        model.eval()
        total, correct = 0, 0
        losses = []

        for images, labels in self.test_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            with torch.no_grad():
                outputs = model(images)
                loss = self.criterion(outputs, labels)
            losses.append(loss.item())
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        return correct / total, sum(losses) / len(losses)

    def test_inference(self, args, model, test_dataset):
        """
        A second test pass over the full test_dataset (outside of the split).
        Returns (accuracy, loss).
        """
        model.eval()
        device = 'cuda' if args.gpu else 'cpu'
        criterion = nn.NLLLoss().to(device)

        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        total, correct = 0, 0
        losses = []

        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs, labels)
            losses.append(loss.item())
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        return correct / total, sum(losses) / len(losses)
