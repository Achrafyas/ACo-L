from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Simple feedforward neural network (multi-layer perceptron).
    - Flatten the input image.
    - One hidden layer with ReLU and dropout.
    - Softmax output for classification.
    """
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        # Input → hidden
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        # Hidden → output
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        # Convert logits to probabilities
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Flatten batch of images to vectors
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        # Linear → Dropout → ReLU → Linear → Softmax
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)


class CNNCustom(nn.Module):
    """
    Example custom CNN with:
    - Two convolutional layers + pooling
    - Three fully connected layers
    - Log-softmax output
    """
    def __init__(self, args):
        super(CNNCustom, self).__init__()
        # conv1 → pool → conv2 → pool
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        # Conv → ReLU → Pool
        x = self.pool(F.relu(self.conv1(x)))
        # Conv → ReLU → Pool
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten
        x = x.view(-1, 16 * 5 * 5)
        # FC → ReLU → FC → ReLU → FC → LogSoftmax
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class CNNMnist(nn.Module):
    """
    CNN for MNIST (single-channel) classification:
    - Two conv layers with max-pooling and dropout
    - Two fully connected layers
    - Log-softmax output
    """
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1       = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2       = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop  = nn.Dropout2d()
        self.fc1         = nn.Linear(320, 50)
        self.fc2         = nn.Linear(50, args.num_classes)

    def forward(self, x):
        # Conv1 → ReLU → Pool
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # Conv2 → Dropout → ReLU → Pool
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # Flatten
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        # FC1 → ReLU → Dropout → FC2 → LogSoftmax
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNNFashion_Mnist(nn.Module):
    """
    CNN for Fashion-MNIST:
    - Two conv+batchnorm+ReLU+pool blocks
    - Single fully connected output
    """
    def __init__(self, args):
        super(CNNFashion_Mnist, self).__init__()
        # Block 1: conv → BN → ReLU → pool
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Block 2: conv → BN → ReLU → pool
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Final linear classification
        self.fc = nn.Linear(7 * 7 * 32, args.num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)  # flatten
        return self.fc(x)


class CNNCifar(nn.Module):
    """
    CNN for CIFAR-like images (3-channel):
    - Two conv+pool layers
    - Three fully connected layers
    - Log-softmax output
    """
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16 * 5 * 5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class AllConvNet(nn.Module):
    """
    Deeper all-convolutional network:
    - Multiple conv layers with occasional down-sampling
    - Final 1×1 conv to number of classes, then global average pooling
    """
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(AllConvNet, self).__init__()
        # Sequence of convolutional layers
        self.conv1      = nn.Conv2d(input_size, 96, 3, padding=1)
        self.conv2      = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3      = nn.Conv2d(96, 96, 3, padding=1, stride=2)  # downsample
        self.conv4      = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5      = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6      = nn.Conv2d(192, 192, 3, padding=1, stride=2)  # downsample
        self.conv7      = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8      = nn.Conv2d(192, 192, 1)
        # Project to class logits
        self.class_conv = nn.Conv2d(192, n_classes, 1)

    def forward(self, x):
        x = F.dropout(x, .2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.dropout(x, .5)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.dropout(x, .5)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        # 1x1 conv → global average pool → squeeze to (batch, n_classes)
        x = F.relu(self.class_conv(x))
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x