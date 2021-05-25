import math

import torch.nn as nn
import torch.nn.functional as F

# TODO: have params for size of network?

# dicts of networks at end of file


class SmallMLP(nn.Module):
    """
    A simple MLP, likely not very competitive
    """
    def __init__(self):
        super(SmallMLP, self).__init__()
        # see README for why this is structured weirdly
        self.hidden1 = 512
        self.hidden2 = 512
        self.hidden3 = 512
        self.layer1 = nn.ModuleDict({"fc": nn.Linear(28 * 28, self.hidden1)})
        self.layer2 = nn.ModuleDict(
            {"fc": nn.Linear(self.hidden1, self.hidden2)})
        self.layer3 = nn.ModuleDict(
            {"fc": nn.Linear(self.hidden2, self.hidden3)})
        self.layer4 = nn.ModuleDict({"fc": nn.Linear(self.hidden3, 10)})

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.layer1["fc"](x))
        x = F.relu(self.layer2["fc"](x))
        x = F.relu(self.layer3["fc"](x))
        x = self.layer4["fc"](x)
        return x


class TinyMLP(nn.Module):
    """
    A tiny MLP used for debugging. Use with a fake tiny dataset.
    """
    def __init__(self):
        super(TinyMLP, self).__init__()
        self.layer1 = nn.ModuleDict({"fc": nn.Linear(3, 3)})
        self.layer2 = nn.ModuleDict({"fc": nn.Linear(3, 2)})

    def forward(self, x):
        x = F.relu(self.layer1["fc"](x))
        x = self.layer2["fc"](x)
        return x


class SmallCNN(nn.Module):
    """
    A simple CNN, taken from the KMNIST benchmark:
    https://github.com/rois-codh/kmnist/blob/master/benchmarks/kuzushiji_mnist_cnn.py
    """
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.hidden1 = 32
        self.hidden2 = 64
        self.hidden3 = 128
        self.hidden4 = 256
        # NOTE: conv layers MUST have names starting with 'conv'
        self.layer1 = nn.ModuleDict({"conv": nn.Conv2d(1, self.hidden1, 3)})
        self.layer2 = nn.ModuleDict({
            "conv":
            nn.Conv2d(self.hidden1, self.hidden2, 3),
            "maxPool":
            nn.MaxPool2d(2, 2),
            "dropout":
            nn.Dropout(p=0.25)
        })
        self.layer3 = nn.ModuleDict(
            {"conv": nn.Conv2d(self.hidden2, self.hidden3, 3)})
        self.layer4 = nn.ModuleDict({
            "fc":
            nn.Linear(self.hidden3 * 10 * 10, self.hidden4),
            "dropout":
            nn.Dropout(p=0.50)
        })
        self.layer5 = nn.ModuleDict({"fc": nn.Linear(self.hidden4, 10)})

    def forward(self, x):
        x = F.relu(self.layer1["conv"](x))
        x = F.relu(self.layer2["conv"](x))
        x = self.layer2["dropout"](self.layer2["maxPool"](x))
        x = F.relu(self.layer3["conv"](x))
        x = x.view(-1, self.num_flat_features(x))
        x = self.layer4["dropout"](F.relu(self.layer4["fc"](x)))
        x = self.layer5["fc"](x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        return math.prod(size)


# TODO: pass in layer widths etc.
class CIFAR10_BN_CNN(nn.Module):
    """
    A 6-layer CNN using batch norm, sized for CIFAR-10
    """
    def __init__(self):
        super(CIFAR10_BN_CNN, self).__init__()
        # see README for why this is structured weirdly
        self.hidden1 = 64
        self.hidden2 = 128
        self.hidden3 = 128
        self.hidden4 = 256
        self.hidden5 = 256
        self.layer1 = nn.ModuleDict(
            {"conv": nn.Conv2d(3, self.hidden1, 3, padding=1)})
        self.layer2 = nn.ModuleDict({
            "conv":
            nn.Conv2d(self.hidden1, self.hidden2, 3, padding=1),
            "maxPool":
            nn.MaxPool2d(2, 2),
            "bn":
            nn.BatchNorm2d(self.hidden2)
        })
        self.layer3 = nn.ModuleDict(
            {"conv": nn.Conv2d(self.hidden2, self.hidden3, 3, padding=1)})
        self.layer4 = nn.ModuleDict({
            "conv":
            nn.Conv2d(self.hidden3, self.hidden4, 3, padding=1),
            "maxPool":
            nn.MaxPool2d(2, 2),
            "bn":
            nn.BatchNorm2d(self.hidden4)
        })
        self.layer5 = nn.ModuleDict({
            "fc":
            nn.Linear(self.hidden4 * 8 * 8, self.hidden5),
            "drop":
            nn.Dropout(p=0.50)
        })
        self.layer6 = nn.ModuleDict({"fc": nn.Linear(self.hidden5, 10)})

    def forward(self, x):
        x = F.relu(self.layer1["conv"](x))
        x = F.relu(self.layer2["bn"](self.layer2["conv"](x)))
        x = self.layer2["maxPool"](x)
        x = F.relu(self.layer3["conv"](x))
        x = F.relu(self.layer4["bn"](self.layer4["conv"](x)))
        x = self.layer4["maxPool"](x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.layer5["drop"](F.relu(self.layer5["fc"](x)))
        x = self.layer6["fc"](x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        return math.prod(size)


class CIFAR10_CNN(nn.Module):
    """
    A 6-layer CNN sized for CIFAR-10
    """
    def __init__(self):
        super(CIFAR10_CNN, self).__init__()
        # see README for why this is structured weirdly
        self.hidden1 = 64
        self.hidden2 = 128
        self.hidden3 = 128
        self.hidden4 = 256
        self.hidden5 = 256
        self.layer1 = nn.ModuleDict(
            {"conv": nn.Conv2d(3, self.hidden1, 3, padding=1)})
        self.layer2 = nn.ModuleDict({
            "conv":
            nn.Conv2d(self.hidden1, self.hidden2, 3, padding=1),
            "maxPool":
            nn.MaxPool2d(2, 2)
        })
        self.layer3 = nn.ModuleDict(
            {"conv": nn.Conv2d(self.hidden2, self.hidden3, 3, padding=1)})
        self.layer4 = nn.ModuleDict({
            "conv":
            nn.Conv2d(self.hidden3, self.hidden4, 3, padding=1),
            "maxPool":
            nn.MaxPool2d(2, 2)
        })
        self.layer5 = nn.ModuleDict({
            "fc":
            nn.Linear(self.hidden4 * 8 * 8, self.hidden5),
            "drop":
            nn.Dropout(p=0.50)
        })
        self.layer6 = nn.ModuleDict({"fc": nn.Linear(self.hidden5, 10)})

    def forward(self, x):
        x = F.relu(self.layer1["conv"](x))
        x = F.relu(self.layer2["conv"](x))
        x = self.layer2["maxPool"](x)
        x = F.relu(self.layer3["conv"](x))
        x = F.relu(self.layer4["conv"](x))
        x = self.layer4["maxPool"](x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.layer5["drop"](F.relu(self.layer5["fc"](x)))
        x = self.layer6["fc"](x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        return math.prod(size)


mlp_dict = {'small': SmallMLP, 'tiny': TinyMLP}

cnn_dict = {
    'cifar10_bn': CIFAR10_BN_CNN,
    'cifar10': CIFAR10_CNN,
    'small': SmallCNN
}
