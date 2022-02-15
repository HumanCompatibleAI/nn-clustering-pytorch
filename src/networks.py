import functools
import math
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F

# TODO: have params for size of network?


class CachingNet(nn.Module):
    """
    Super-class for networks that will cache their activations. For use with
    activation-aware clustering.
    Upon initializing nets, be sure to add a layer_dict and run
    self.cache_activations() after initializing everything else.
    Only works for MLPs at the moment.
    Because it saves the results of fc modules, it doesn't take batchnorm into
    acccount.
    """
    def __init__(self):
        super(CachingNet, self).__init__()
        self.layer_dict = {}
        self.sequential_block = None
        # TODO: figure out how layer_dict works for nets constructed w/
        # sequential modules.

    def cache_activations(self):
        """
        Creates forward hooks to cache the pre-ReLU activations for each layer.
        """
        self.activations = {}

        if self.sequential_block is not None:
            seq_block = getattr(self, self.sequential_block)
            for name, module in seq_block.named_children():
                if not name.startswith('0_'):
                    self.layer_dict[name] = module

        def hook(model, inp, out, name):
            self.activations[name] = out.detach()

        for name, layer in self.layer_dict.items():
            if hasattr(layer, 'fc'):
                layer.fc.register_forward_hook(
                    functools.partial(hook, name=name))
            elif isinstance(layer, nn.Linear):
                layer.register_forward_hook(functools.partial(hook, name=name))
            elif hasattr(layer, 'conv'):
                layer.conv.register_forward_hook(
                    functools.partial(hook, name=name))
            elif isinstance(layer, nn.Conv2d):
                layer.register_forward_hook(functools.partial(hook, name=name))

        return self.activations


class MnistMLP(CachingNet):
    """
    A simple MLP, likely not very competitive
    """
    def __init__(self):
        super(MnistMLP, self).__init__()
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
        self.layer_dict = {
            'layer1': self.layer1,
            'layer2': self.layer2,
            'layer3': self.layer3,
            'layer4': self.layer4
        }
        self.cache_activations()

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.layer1["fc"](x))
        x = F.relu(self.layer2["fc"](x))
        x = F.relu(self.layer3["fc"](x))
        x = self.layer4["fc"](x)
        return x


class TinyMLP(CachingNet):
    """
    A tiny MLP used for debugging. Use with a fake tiny dataset.
    """
    def __init__(self):
        super(TinyMLP, self).__init__()
        self.layer1 = nn.ModuleDict({"fc": nn.Linear(3, 3)})
        self.layer2 = nn.ModuleDict({"fc": nn.Linear(3, 2)})
        self.layer_dict = {'layer1': self.layer1, 'layer2': self.layer2}
        self.cache_activations()

    def forward(self, x):
        x = F.relu(self.layer1["fc"](x))
        x = self.layer2["fc"](x)
        return x


class AddMulMLP(CachingNet):
    """
    An MLP used for the addition/multiplication task from Csordas (2021).
    """
    def __init__(self):
        super(AddMulMLP, self).__init__()
        self.layer1 = nn.ModuleDict({"fc": nn.Linear(42, 2000)})
        self.layer2 = nn.ModuleDict({"fc": nn.Linear(2000, 2000)})
        self.layer3 = nn.ModuleDict({"fc": nn.Linear(2000, 2000)})
        self.layer4 = nn.ModuleDict({"fc": nn.Linear(2000, 2000)})
        self.layer5 = nn.ModuleDict({"fc": nn.Linear(2000, 20)})
        self.layer_dict = {
            'layer1': self.layer1,
            'layer2': self.layer2,
            'layer3': self.layer3,
            'layer4': self.layer4,
            'layer5': self.layer5
        }
        self.cache_activations()

    def forward(self, x):
        x = F.relu(self.layer1["fc"](x))
        x = F.relu(self.layer2["fc"](x))
        x = F.relu(self.layer3["fc"](x))
        x = F.relu(self.layer4["fc"](x))
        x = self.layer5["fc"](x)
        return x


class SimpleMathMLP(CachingNet):
    """
    An MLP used to model simple math functions, eg (x,y) -> (sin(x), cos(y))
    4 layer MLP, with 3 hidden layers of the same size

    out (int): The number of outputs
    input_type (str) : "single" means there is a single input, and the output
        is expected to be (sin(x), cos(x)). "streamed" means there are as many
        inputs as outputs and a 1-1 mapping, eg (x,y)->(sin(x),cos(y))
    hidden (int): The size of the hidden layers
    """
    def __init__(self, out=2, input_type="single", hidden=512):
        super(SimpleMathMLP, self).__init__()
        self.input_type = input_type
        if input_type == "single":
            inp = 1
        elif input_type == "streamed":
            inp = out
        self.inp = inp
        self.out = out
        self.hidden1 = hidden
        self.hidden2 = hidden
        self.hidden3 = hidden
        self.layer1 = nn.ModuleDict({"fc": nn.Linear(inp, self.hidden1)})
        self.layer2 = nn.ModuleDict(
            {"fc": nn.Linear(self.hidden1, self.hidden2)})
        self.layer3 = nn.ModuleDict(
            {"fc": nn.Linear(self.hidden2, self.hidden3)})
        self.layer4 = nn.ModuleDict({"fc": nn.Linear(self.hidden3, out)})
        self.layer_dict = {
            'layer1': self.layer1,
            'layer2': self.layer2,
            'layer3': self.layer3,
            'layer4': self.layer4
        }
        self.cache_activations()

    def forward(self, x):
        if len(x.shape) == 1:
            x = x[:, None]
        act = F.relu
        x = act(self.layer1["fc"](x))
        x = act(self.layer2["fc"](x))
        x = act(self.layer3["fc"](x))
        x = self.layer4["fc"](x)
        return x


class MnistCNN(CachingNet):
    """
    A simple CNN, taken from the KMNIST benchmark:
    https://github.com/rois-codh/kmnist/blob/master/benchmarks/kuzushiji_mnist_cnn.py
    """
    def __init__(self):
        super(MnistCNN, self).__init__()
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
        self.layer_dict = {'layer2': self.layer2, 'layer3': self.layer3}
        self.cache_activations()

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
class CIFAR10_BN_CNN_6(CachingNet):
    """
    A 6-layer CNN using batch norm, sized for CIFAR-10
    """
    def __init__(self):
        super(CIFAR10_BN_CNN_6, self).__init__()
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
        self.layer_dict = {
            'layer2': self.layer2,
            'layer3': self.layer3,
            'layer4': self.layer4
        }
        self.cache_activations()

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


class CIFAR10_CNN_6(CachingNet):
    """
    A 6-layer CNN sized for CIFAR-10
    """
    def __init__(self):
        super(CIFAR10_CNN_6, self).__init__()
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
        self.layer_dict = {
            'layer2': self.layer2,
            'layer3': self.layer3,
            'layer4': self.layer4
        }
        self.cache_activations()

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


class CIFAR10_VGG(CachingNet):
    """
    Code to generate VGGs for CIFAR-10 (together with make_layers). Modified
    from
    https://github.com/chengyangfu/pytorch-vgg-cifar10/blob/master/vgg.py
    as well as
    https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
    """
    def __init__(self, conv_features, init_weights=True):
        super(CIFAR10_VGG, self).__init__()
        self.conv_features = conv_features
        self.fc_ord_dict = OrderedDict([("0_dropout", nn.Dropout()),
                                        ("1_fc", nn.Linear(512, 512)),
                                        ("1_relu", nn.ReLU()),
                                        ("1_dropout", nn.Dropout()),
                                        ("2_fc", nn.Linear(512, 512)),
                                        ("2_relu", nn.ReLU()),
                                        ("3_fc", nn.Linear(512, 10))])
        self.fc_classifier = nn.Sequential(self.fc_ord_dict)
        if init_weights:
            self._initialize_weights()
        self.sequential_block = 'conv_features'
        self.cache_activations()

    def forward(self, x):
        x = self.conv_features(x)
        x = x.view(x.size(0), -1)
        x = self.fc_classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal(m.weight, 0, 0.01)
                    nn.init.constant(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = OrderedDict()
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers[f"{i}_maxPool"] = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            layers[f"{i}_conv"] = nn.Conv2d(in_channels,
                                            v,
                                            kernel_size=3,
                                            padding=1)
            if batch_norm:
                layers[f"{i}_bn"] = nn.BatchNorm2d(v)
            layers[f"{i}_relu"] = nn.ReLU()
            if cfg[i + 1] != 'M':
                dropout_rate = 0.3 if i != 0 else 0.4
                layers[f"{i}_dropout"] = nn.Dropout(dropout_rate)
            elif i == len(cfg) - 2:
                layers[f"{i}_dropout"] = nn.Dropout(0.5)
            in_channels = v
    return nn.Sequential(layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B':
    [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
        512, 512, 512, 'M'
    ],
    'E': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512,
        512, 'M', 512, 512, 512, 512, 'M'
    ],
}


def cifar10_vgg11():
    """VGG 11-layer model (configuration "A")"""
    return CIFAR10_VGG(make_layers(cfg['A']))


def cifar10_vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return CIFAR10_VGG(make_layers(cfg['A'], batch_norm=True))


def cifar10_vgg13():
    """VGG 13-layer model (configuration "B")"""
    return CIFAR10_VGG(make_layers(cfg['B']))


def cifar10_vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return CIFAR10_VGG(make_layers(cfg['B'], batch_norm=True))


def cifar10_vgg16():
    """VGG 16-layer model (configuration "D")"""
    return CIFAR10_VGG(make_layers(cfg['D']))


def cifar10_vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return CIFAR10_VGG(make_layers(cfg['D'], batch_norm=True))


def cifar10_vgg19():
    """VGG 19-layer model (configuration "E")"""
    return CIFAR10_VGG(make_layers(cfg['E']))


def cifar10_vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return CIFAR10_VGG(make_layers(cfg['E'], batch_norm=True))


MLP_DICT = {
    'mnist': MnistMLP,
    'tiny': TinyMLP,
    'add_mul': AddMulMLP,
    'simple': SimpleMathMLP
}

CNN_DICT = {
    'cifar10_6_bn': CIFAR10_BN_CNN_6,
    'cifar10_6': CIFAR10_CNN_6,
    'mnist': MnistCNN,
    'cifar10_vgg11': cifar10_vgg11,
    'cifar10_vgg11_bn': cifar10_vgg11_bn,
    'cifar10_vgg13': cifar10_vgg13,
    'cifar10_vgg13_bn': cifar10_vgg13_bn,
    'cifar10_vgg16': cifar10_vgg16,
    'cifar10_vgg16_bn': cifar10_vgg16_bn,
    'cifar10_vgg19': cifar10_vgg19,
    'cifar10_vgg19_bn': cifar10_vgg19_bn
}
