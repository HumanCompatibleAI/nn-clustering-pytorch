import collections
import time

import numpy as np
import torch


def get_random_int_time():
    """
    get a random int based on the least significant part of the time
    """
    time_str = str(time.time())
    dcml_place = time_str.index('.')
    return int(time_str[dcml_place + 1:])


def compute_percentile(x, arr):
    r = np.sum(arr < x)
    n = len(arr)
    return r / n


def get_weighty_modules_from_live_net(network):
    """
    Takes a neural network, and returns the modules from it that have proper
    weight tensors (i.e. not including batchnorm modules)
    NB: so far, requires things to only have linear layers
    network: a neural network, that has to inherit from nn.Module
    returns: a list of nn.Modules
    """
    weighty_modules = []
    for module in network.modules():
        if isinstance(module, torch.nn.Linear):
            weighty_modules.append(module)
    return weighty_modules


def get_graph_weights_from_live_net(network):
    """
    Takes a neural network, and gets weights from it to use to turn it into a
    graph.
    NB: requires things to only have linear layers
    network: a neural network, currently an MLP. Probably has to inherit from
             nn.Module
    returns: a list of pytorch tensors.
    """
    weight_tensors = []
    for module in network.modules():
        if isinstance(module, torch.nn.Linear):
            weight_tensors.append(module.weight)
    return weight_tensors


def get_graph_weights_from_state_dict(state_dict):
    """
    Takes a pytorch state dict, and returns an array of the weight tensors that
    constitute the graph we're working with.
    NB: relies on the dict having the expected order.
    NB: also relies on the network not being actively pruned
    NB: might break once batch norm happens
    state_dict: a pytorch state dict
    returns: an array of pytorch tensors
    """
    assert isinstance(state_dict, collections.OrderedDict)
    weights = []
    for string in state_dict:
        if string.endswith("weight"):
            weights.append(state_dict[string])
    return weights


def load_model_weights_pytorch(model_path, pytorch_device):
    """
    Take a pytorch saved model state dict, and return an array of the weight
    tensors as numpy arrays
    NB: this relies on the dict being ordered in the right order.
    model_path: a string, pointing to a pytorch saved state_dict
    pytorch_device: pytorch device, which device to save the model to
    returns: array of numpy arrays of weight tensors (no biases)
    """
    state_dict = torch.load(model_path, map_location=pytorch_device)
    torch_weights = get_graph_weights_from_state_dict(state_dict)
    np_weights = [tens.detach().cpu().numpy() for tens in torch_weights]
    return np_weights


def weights_to_layer_widths(weights_array):
    """
    take in an array of weight matrices, and return how wide each layer of the
    network is
    weights_array: an array of numpy arrays representing NN layer tensors
    Returns a list of ints, each representing the width of a layer.
    """
    for i in range(len(weights_array) - 1):
        assert weights_array[i].shape[0] == weights_array[i + 1].shape[1]
    layer_widths = [x.shape[1] for x in weights_array]
    layer_widths.append(weights_array[-1].shape[0])
    return layer_widths
