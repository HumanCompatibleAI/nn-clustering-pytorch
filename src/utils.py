import collections
import time

import numpy as np
import torch

net_types = ['mlp', 'cnn']


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
    weighty_module_types = [torch.nn.Linear, torch.nn.Conv2d]
    for module in network.modules():
        if any([isinstance(module, t) for t in weighty_module_types]):
            weighty_modules.append(module)
    return weighty_modules


def get_graph_weights_from_live_net(network, net_type):
    """
    Takes a neural network, and gets weights from it to use to turn it into a
    graph.
    NB: for conv net, assumes all conv layers are contiguous.
    network: a neural network. Has to inherit from nn.Module.
    net_type: string indicating whether the net is an MLP or a CNN
    returns: a list of pytorch tensors.
    """
    assert net_type in net_types
    weight_tensors = []
    if net_type == 'mlp':
        for module in network.modules():
            if isinstance(module, torch.nn.Linear):
                weight_tensors.append(module.weight)
    elif net_type == 'cnn':
        # TODO: deal with case where you have linear layers before conv layers
        for i, module in enumerate(network.modules()):
            if i == 0:
                # we don't have the inputs as part of the graph in convnets
                continue
            elif isinstance(module, torch.nn.Conv2d):
                # we only include convolutional layers
                weight_tensors.append(module.weight)
    return weight_tensors


def get_graph_weights_from_state_dict(state_dict, net_type):
    """
    Takes a pytorch state dict, and returns an array of the weight tensors that
    constitute the graph we're working with.
    NB: relies on the dict having the expected order.
    NB: also relies on the network not being actively pruned
    NB: also relies on all conv layers being contiguous and having names
        starting with 'conv'
    NB: might break once batch norm happens
    state_dict: a pytorch state dict
    net_type: string indicating whether the net is an MLP or a CNN
    returns: an array of pytorch tensors
    """
    assert net_type in net_types
    assert isinstance(state_dict, collections.OrderedDict)
    weights = []
    if net_type == 'mlp':
        for string in state_dict:
            if string.endswith("weight"):
                weights.append(state_dict[string])
    elif net_type == 'cnn':
        # TODO: deal with case where you have linear layers before conv layers
        for (i, string) in enumerate(state_dict):
            if i == 0:
                # don't include the inputs as part of the graph in conv nets
                continue
            elif string.startswith("conv") and string.endswith("weight"):
                # add weights of conv layers
                weights.append(state_dict[string])
    return weights


def load_model_weights_pytorch(model_path, net_type, pytorch_device):
    """
    Take a pytorch saved model state dict, and return an array of the weight
    tensors as numpy arrays
    NB: this relies on the dict being ordered in the right order.
    model_path: a string, pointing to a pytorch saved state_dict
    net_type: string indicating whether the model is an MLP or a CNN
    pytorch_device: pytorch device, which device to save the model to
    returns: array of numpy arrays of weight tensors (no biases)
    """
    state_dict = torch.load(model_path, map_location=pytorch_device)
    torch_weights = get_graph_weights_from_state_dict(state_dict, net_type)
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


def vector_stretch(vector, length):
    """
    takes a pytorch 1d tensor, and 'stretches' it so that it has the right
    length. e.g. vector_stretch(torch.Tensor([1,2,3]), 6)
    == torch.Tensor([1,1,2,2,3,3]).
    vector: 1d pytorch tensor
    length: int length to stretch to. Must be an integer multiple of the length
            of vector
    returns: 1d pytorch tensor.
    """
    assert isinstance(vector, torch.Tensor)
    assert len(vector.shape) == 1
    start_len = vector.shape[0]
    assert length % start_len == 0
    mult = int(length / start_len)
    stretched_vec = torch.empty(length)
    for i in range(start_len):
        val = vector[i]
        for j in range(mult):
            stretched_vec[i * mult + j] = val
    return stretched_vec
