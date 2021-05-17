import itertools
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

from clusterability_gradient import LaplacianEigenvalues
from utils import (
    get_weight_modules_from_live_net,
    size_and_multiply_np,
    size_sqrt_divide_np,
    vector_stretch,
)

train_exp = Experiment('train_model')
train_exp.captured_out_filter = apply_backspaces_and_linefeeds
train_exp.observers.append(FileStorageObserver('training_runs'))

# probably should define a global variable for the list of datasets.
# maybe in a utils file or something.

# TODO: have a function that tells you the dimensions of the stuff in your
# dataset


@train_exp.config
def mlp_config():
    batch_size = 128
    num_epochs = 20
    log_interval = 100
    dataset = 'kmnist'
    model_dir = './models/'
    net_type = 'mlp'
    net_choice = 'small'
    # pruning will be as described in Zhu and Gupta 2017, arXiv:1710.01878
    pruning_config = {
        'exponent': 3,
        'frequency': 100,
        'num pruning epochs': 5,
        # no pruning if num pruning epochs = 0
        'final sparsity': 0.9
    }
    cluster_gradient = False
    cluster_gradient_config = {
        'num_workers': 2,
        'num_eigs': 3,
        'lambda': 1e-3,
        'frequency': 20,
        'normalize': True
    }
    save_path_prefix = (model_dir + net_type + '_' + dataset +
                        cluster_gradient * '_clust-grad')
    # TODO: figure out what info should go in here.
    _ = locals()
    del _


@train_exp.named_config
def cnn_config():
    net_type = 'cnn'
    _ = locals()
    del _


@train_exp.capture
def load_datasets(dataset, batch_size):
    """
    get loaders for training datasets, as well as a description of the classes.
    dataset: string representing the dataset.
    return pytorch loader for training set, pytorch loader for test set,
    tuple of names of classes.
    """
    assert dataset in ['kmnist', 'cifar10']
    if dataset == 'kmnist':
        return load_kmnist(batch_size)
    elif dataset == 'cifar10':
        return load_cifar10(batch_size)
    else:
        raise ValueError("Wrong name for dataset!")


def load_kmnist(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    train_set = torchvision.datasets.KMNIST(root="./datasets",
                                            train=True,
                                            download=True,
                                            transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_set = torchvision.datasets.KMNIST(root="./datasets",
                                           train=False,
                                           download=True,
                                           transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=True)
    classes = ("o", "ki", "su", "tsu", "na", "ha", "ma", "ya", "re", "wo")
    return (train_loader, test_loader, classes)


def load_cifar10(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_set = torchvision.datasets.CIFAR10(root="./datasets",
                                             train=True,
                                             download=True,
                                             transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_set = torchvision.datasets.CIFAR10(root="./datasets",
                                            train=False,
                                            download=True,
                                            transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=True)
    classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse",
               "ship", "truck")
    return train_loader, test_loader, classes


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


mlp_dict = {'small': SmallMLP}


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


cnn_dict = {
    'cifar10_bn': CIFAR10_BN_CNN,
    'cifar10': CIFAR10_CNN,
    'small': SmallCNN
}


def module_array_to_clust_grad_input(weight_modules, net_type):
    """
    Turns a layer_array of modules into objects that can be nicely fed into
    LaplacianEigenvalues.
    weight_modules: list of dicts containing a layer name and a variety of
                    pytorch modules
    net_type: string specifying whether the network is a CNN or an MLP
    Returns: tuple of (list of pytorch tensors, list of strings specifying what
                                                each tensor is)
    """
    tensor_array = []
    tensor_type_array = []
    assert net_type in ['mlp', 'cnn']
    weight_module_name, weight_name = (('fc_mod',
                                        'fc_weights') if net_type == 'mlp' else
                                       ('conv_mod', 'conv_weights'))

    def has_weights(my_dict):
        return weight_module_name in my_dict

    for k, g in itertools.groupby(weight_modules, has_weights):
        if k:
            weight_layers = list(g) if net_type == 'mlp' else list(g)[1:]
            break

    for layer_dict in weight_layers:
        tensor_array.append(layer_dict[weight_module_name].weight)
        tensor_type_array.append(weight_name)
        if 'bn_mod' in layer_dict:
            bn_mod = layer_dict['bn_mod']
            if hasattr(bn_mod, 'weight') and bn_mod.weight is not None:
                tensor_array.append(bn_mod.weight)
                tensor_type_array.append('bn_weights')
            tensor_array.append(bn_mod.running_var)
            tensor_type_array.append('bn_running_var')
    return tensor_array, tensor_type_array


def calculate_clust_reg(cluster_gradient_config, net_type, network):
    """
    Calculate the clusterability regularization term of a network.
    cluster_gradient_config: dict containing 'num_eigs', the int number of
                             eigenvalues to regularize, 'num_workers', the int
                             number of CPU workers to use to calculate the
                             gradient, 'lambda', the float regularization
                             strength to use per eigenvalue, and 'frequency',
                             the number of iterations between successive
                             applications of the term.
    net_type: string indicating whether the network is an MLP or a CNN
    network: a pytorch network.
    returns: a tensor float.
    """
    num_workers = cluster_gradient_config['num_workers']
    num_eigs = cluster_gradient_config['num_eigs']
    cg_lambda = cluster_gradient_config['lambda']
    weight_modules = get_weight_modules_from_live_net(network)
    tensor_arrays = module_array_to_clust_grad_input(weight_modules, net_type)
    tensor_array, tensor_type_array = tensor_arrays
    eig_sum = torch.sum(
        LaplacianEigenvalues.apply(num_workers, num_eigs, net_type,
                                   tensor_type_array, *tensor_array))
    return (cg_lambda / num_eigs) * eig_sum


def normalize_weights(network, eps=1e-3):
    """
    'Normalize' the weights of a network, so that for each hidden neuron, the
    norm of incoming weights to that neuron is sqrt(2). For a ReLU network,
    this operation preserves network functionality.
    network: a neural network. has to inherit from torch.nn.module. Currently
             probably has to be an MLP
    eps: a float that should be small relative to sqrt(2), to add stability.
    returns nothing: just modifies the network in-place
    """
    # TODO: figure out whether I should be normalizing BN layers
    layers = get_weight_modules_from_live_net(network)
    for idx in range(len(layers) - 1):
        this_layer = layers[idx]
        next_layer = layers[idx + 1]
        assert 'fc_mod' in this_layer or 'conv_mod' in this_layer
        assert 'fc_mod' in next_layer or 'conv_mod' in next_layer
        inc_raw_weight_mod = (this_layer['fc_mod'] if 'fc_mod' in this_layer
                              else this_layer['conv_mod'])
        inc_raw_weights = inc_raw_weight_mod.weight
        inc_raw_bias = inc_raw_weight_mod.bias
        inc_weights_np = inc_raw_weights.detach().cpu().numpy()
        inc_biases_np = inc_raw_bias.detach().cpu().numpy()
        if 'bn_mod' in this_layer:
            bn_mod = this_layer['bn_mod']
            if hasattr(bn_mod, 'weight') and bn_mod.weight is not None:
                bn_weights_np = bn_mod.weight.detach().cpu().numpy()
                inc_weights_np = size_and_multiply_np(bn_weights_np,
                                                      inc_weights_np)
            inc_weights_np = size_sqrt_divide_np(bn_mod.running_var,
                                                 inc_weights_np)
        outgoing_weight_mod = (next_layer['fc_mod'] if 'fc_mod' in next_layer
                               else next_layer['conv_mod'])
        outgoing_weights = outgoing_weight_mod.weight
        num_neurons = inc_weights_np.shape[0]
        assert outgoing_weights.shape[1] % num_neurons == 0
        if 'fc_mod' in this_layer and 'fc_mod' in next_layer:
            assert outgoing_weights.shape[1] == num_neurons
        if 'conv_mod' in this_layer and 'conv_mod' in next_layer:
            assert outgoing_weights.shape[1] == num_neurons

        unsqueezed_bias = np.expand_dims(inc_biases_np, axis=1)
        flat_weights = inc_weights_np.reshape(inc_weights_np.shape[0], -1)
        all_inc_weights = np.concatenate((flat_weights, unsqueezed_bias),
                                         axis=1)
        scales = np.linalg.norm(all_inc_weights, axis=1)
        scales /= np.sqrt(2.)
        scales += eps
        scales = torch.from_numpy(scales)
        scales_rows = torch.unsqueeze(scales, 1)
        for i in range(len(inc_raw_weights.shape)):
            if i > 1:
                scales_rows = torch.unsqueeze(scales_rows, i)
        scales_mul = vector_stretch(scales, outgoing_weights.shape[1])
        for i in range(len(outgoing_weights.shape) - 1):
            if i > 0:
                scales_mul = torch.unsqueeze(scales_mul, i)

        incoming_weights_unpruned = True
        incoming_biases_unpruned = True
        outgoing_weights_unpruned = True
        for name, param in inc_raw_weight_mod.named_parameters():
            if name == 'weight_orig':
                param.data = torch.div(param, scales_rows)
                incoming_weights_unpruned = False
            if name == 'bias_orig':
                param.data = torch.div(param, scales)
                incoming_biases_unpruned = False
        for name, param in outgoing_weight_mod.named_parameters():
            if name == 'weight_orig':
                param.data = torch.mul(param, scales_mul)
                outgoing_weights_unpruned = False
        if incoming_weights_unpruned:
            inc_raw_weight_mod.weight.data = torch.div(inc_raw_weights,
                                                       scales_rows)
        if incoming_biases_unpruned:
            inc_raw_weight_mod.bias.data = torch.div(inc_raw_bias, scales)
        if outgoing_weights_unpruned:
            outgoing_weight_mod.weight.data = torch.mul(
                outgoing_weights, scales_mul)


def calculate_sparsity_factor(final_sparsity, num_prunes_so_far,
                              num_prunes_total, prune_exp, current_density):
    """
    Convenience function to calculate the sparsity factor to prune with.
    final_sparsity: a float between 0 and 1
    num_prunes_so_far: an integer for the number of times pruning has been
                       applied.
    num_prunes_total: an integer representing the total number of times pruning
                      will ever occur.
    prune_exp: a numeric type regulating the pruning schedule
    current_density: a float representing how dense the network currently is.
    returns: a float giving the factor of weights to prune.
    """
    assert final_sparsity >= 0
    assert final_sparsity <= 1
    assert current_density >= 1 - final_sparsity
    # is the above actually exactly true?
    assert current_density <= 1
    new_sparsity = (final_sparsity *
                    (1 - (1 -
                          (num_prunes_so_far / num_prunes_total))**prune_exp))
    sparsity_factor = 1 - ((1. - new_sparsity) / current_density)
    return sparsity_factor


def get_prunable_modules(network):
    """
    Takes a neural network, and returns the modules from it that can be pruned.
    network: a neural network, that has to inherit from nn.Module
    returns: a list of nn.Modules
    """
    prunable_modules = []
    prunable_module_types = [torch.nn.Linear, torch.nn.Conv2d]
    for module in network.modules():
        if any([isinstance(module, t) for t in prunable_module_types]):
            prunable_modules.append(module)
    return prunable_modules


def train_and_save(network, net_type, train_loader, test_loader, num_epochs,
                   pruning_config, cluster_gradient, cluster_gradient_config,
                   optimizer, criterion, log_interval, device,
                   model_path_prefix, _run):
    """
    Train a neural network, printing out log information, and saving along the
    way.
    network: an instantiated object that inherits from nn.Module or something.
    net_type: string indicating whether the net is an MLP or a CNN
    train_loader: pytorch loader for the training dataset
    test_loader: pytorch loader for the testing dataset
    num_epochs: int for the total number of epochs to train for (including any
                pruning)
    pruning_config: dict containing 'exponent', a numeric type, 'frequency',
                    int representing the number of training steps to have
                    between prunes, 'num pruning epochs', int representing the
                    number of epochs to prune for, and 'final sparsity', float
                    representing how sparse the final net should be.
                    If 'num pruning epochs' is 0, no other elements are
                    accessed.
    cluster_gradient: bool representing whether or not to apply the
                      clusterability gradient
    cluster_gradient_config: dict containing 'num_eigs', the int number of
                             eigenvalues to regularize, 'num_workers', the int
                             number of CPU workers to use to calculate the
                             gradient, 'lambda', the float regularization
                             strength to use per eigenvalue, and 'frequency',
                             the number of iterations between successive
                             applications of the term. Only accessed if
                             cluster_gradient is True.
    optimizer: pytorch optimizer. Might be SGD or Adam.
    criterion: loss function.
    log_interval: int. how many training steps between logging infodumps.
    device: pytorch device - do things go on CPU or GPU?
    model_path_prefix: string containing the relative path to save the model.
                       should not include final '.pth' suffix.
    returns: tuple of test acc, test loss, and list of tuples of
             (epoch number, iteration number, train loss)
    """
    network.to(device)
    loss_list = []

    num_pruning_epochs = pruning_config['num pruning epochs']
    final_sparsity = pruning_config['final sparsity']
    start_pruning_epoch = num_epochs - num_pruning_epochs
    is_pruning = num_pruning_epochs != 0
    if is_pruning:
        prune_exp = pruning_config['exponent']
        prune_freq = pruning_config['frequency']
        current_density = 1.
        num_prunes_total = ((num_pruning_epochs * len(train_loader)) //
                            prune_freq)
        num_prunes_so_far = 0
        train_step_counter = 0

    prune_modules_list = get_prunable_modules(network)
    prune_params_list = [(mod, 'weight') for mod in prune_modules_list]

    for epoch in range(num_epochs):
        print("Start of epoch", epoch)
        network.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            if cluster_gradient:
                if i % cluster_gradient_config['frequency'] == 0:
                    if cluster_gradient_config['normalize']:
                        normalize_weights(network)
                    clust_reg_term = calculate_clust_reg(
                        cluster_gradient_config, net_type, network)
                    loss += (clust_reg_term *
                             cluster_gradient_config['frequency'])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % log_interval == log_interval - 1:
                avg_loss = running_loss / log_interval
                print(f"batch {i}, avg loss {avg_loss}")
                if cluster_gradient:
                    print(f"cluster regularization term {clust_reg_term}")
                _run.log_scalar("training.loss", avg_loss)
                loss_list.append((epoch, i, avg_loss))
                running_loss = 0.0
            if epoch >= start_pruning_epoch:
                if train_step_counter % prune_freq == 0:
                    sparsity_factor = calculate_sparsity_factor(
                        final_sparsity, num_prunes_so_far, num_prunes_total,
                        prune_exp, current_density)
                    for module, name in prune_params_list:
                        prune.l1_unstructured(module, name, sparsity_factor)
                    current_density *= 1 - sparsity_factor
                    num_prunes_so_far += 1
                train_step_counter += 1

        test_acc, test_loss = eval_net(network, test_loader, device, criterion,
                                       _run)
        print("Test accuracy is", test_acc)
        print("Test loss is", test_loss)
        if is_pruning and epoch == start_pruning_epoch - 1:
            model_path = model_path_prefix + '_unpruned.pth'
            torch.save(network.state_dict(), model_path)
            train_exp.add_artifact(model_path)
            print("Pre-pruning network saved at " + model_path)

    if is_pruning:
        for module, name in prune_params_list:
            prune.remove(module, name)
    save_path = model_path_prefix + '.pth'
    torch.save(network.state_dict(), save_path)
    train_exp.add_artifact(save_path)
    print("Network saved at " + save_path)
    return test_acc, test_loss, loss_list


def eval_net(network, test_loader, device, criterion, _run):
    """
    gets test loss and accuracy
    network: network to get loss of
    test_loader: pytorch loader of test set
    device: device to put data on
    criterion: loss function
    returns: tuple of floats. first is test accuracy, second is test loss.
    """
    correct = 0.0
    total = 0
    loss_sum = 0.0
    num_batches = 0

    network.eval()
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = network(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            loss_sum += loss.item()
            num_batches += 1

    test_accuracy = correct / total
    test_avg_loss = loss_sum / num_batches
    _run.log_scalar("test.accuracy", test_accuracy)
    _run.log_scalar("test.loss", test_avg_loss)
    return test_accuracy, test_avg_loss


@train_exp.automain
def run_training(dataset, net_type, net_choice, num_epochs, batch_size,
                 log_interval, pruning_config, cluster_gradient,
                 cluster_gradient_config, save_path_prefix, _run):
    """
    Trains and saves network.
    dataset: string specifying which dataset we're using
    net_type: string indicating whether the model is an MLP or a CNN
    net_choice: string choosing which model to train
    num_epochs: int
    batch_size: int
    log_interval: int. number of iterations to go between logging infodumps
    pruning_config: dict containing 'exponent', a numeric type, 'frequency',
                    int representing the number of training steps to have
                    between prunes, 'num pruning epochs', int representing the
                    number of epochs to prune for, and 'final sparsity', float
                    representing how sparse the final net should be
    cluster_gradient: bool representing whether or not to apply the
                      clusterability gradient
    cluster_gradient_config: dict containing 'num_eigs', the int number of
                             eigenvalues to regularize, 'num_workers', the int
                             number of CPU workers to use to calculate the
                             gradient, 'lambda', the float regularization
                             strength to use per eigenvalue, and 'frequency',
                             the number of iterations between successive
                             applications of the term. Only accessed if
                             cluster_gradient is True.
    save_path_prefix: string that acts as a prefix for the path where models
                      will be saved to.
    """
    device = (torch.device("cuda")
              if torch.cuda.is_available() else torch.device("cpu"))
    criterion = nn.CrossEntropyLoss()
    my_net = (mlp_dict[net_choice]()
              if net_type == 'mlp' else cnn_dict[net_choice]())
    optimizer = optim.Adam(my_net.parameters())
    train_loader, test_loader, classes = load_datasets()
    test_acc, test_loss, loss_list = train_and_save(
        my_net, net_type, train_loader, test_loader, num_epochs,
        pruning_config, cluster_gradient, cluster_gradient_config, optimizer,
        criterion, log_interval, device, save_path_prefix, _run)
    return {
        'test acc': test_acc,
        'test loss': test_loss,
        'train loss list': loss_list
    }
