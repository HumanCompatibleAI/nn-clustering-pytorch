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


# def get_weighty_modules_from_live_net(network):
#     """
#     Takes a neural network, and returns the modules from it that have proper
#     weight tensors (i.e. not including batchnorm modules)
#     network: a neural network, that has to inherit from nn.Module
#     returns: a list of nn.Modules
#     """
#     weighty_modules = []
#     weighty_module_types = [torch.nn.Linear, torch.nn.Conv2d]
#     for module in network.modules():
#         if any([isinstance(module, t) for t in weighty_module_types]):
#             weighty_modules.append(module)
#     return weighty_modules


def get_weight_modules_from_live_net(network):
    """
    Takes a neural network, and gets modules in it that contain relevant
    weights.
    network: a neural network. Has to inherit from nn.Module.
    returns: an array of dicts containing layer names and relevant pytorch
             modules
    """
    layer_array = []
    for layer_name, layer_mod in network.named_children():
        if isinstance(layer_mod, torch.nn.ModuleDict):
            layer_dict = {'layer': layer_name}
            for mod_name, module in layer_mod.named_children():
                if mod_name == 'fc' and isinstance(module, torch.nn.Linear):
                    layer_dict['fc_mod'] = module
                if mod_name == 'conv' and isinstance(module, torch.nn.Conv2d):
                    layer_dict['conv_mod'] = module
                if (mod_name == 'bn'
                        and (isinstance(module, torch.nn.BatchNorm2d)
                             or isinstance(module, torch.nn.BatchNorm1d))):
                    layer_dict['bn_mod'] = module
            if len(layer_dict) > 1:
                layer_array.append(layer_dict)
    check_layer_names(layer_array)
    return layer_array


def get_weight_tensors_from_state_dict(state_dict):
    """
    Takes a pytorch state dict, and returns an array of the weight tensors that
    constitute the graph we're working with.
    NB: relies on the dict having the expected order.
    NB: also relies on the network not being actively pruned
    NB: also relies on the network having the expected names and structure: see
        README for details
    NB: also relies on MLPs not using batch norm
    state_dict: a pytorch state dict
    returns: an array of dicts containing layer names and various pytorch
             tensors
    """
    assert isinstance(state_dict, collections.OrderedDict)
    layer_array = []
    for name, tens in state_dict.items():
        # these names look like layer4.fc.weight or layer1.bn.running_var
        name_parts = name.split('.')
        assert len(name_parts) == 3, name
        new_layer_name = name_parts[0]
        module_name = name_parts[1]
        attr_name = name_parts[2]
        old_layer = (bool(layer_array)
                     and layer_array[-1]['layer'] == new_layer_name)
        if module_name == "fc" and attr_name == "weight" and not old_layer:
            layer_array.append({'layer': new_layer_name, 'fc_weights': tens})
        if (module_name == "conv" and attr_name == "weight" and not old_layer):
            layer_array.append({'layer': new_layer_name, 'conv_weights': tens})
        if module_name == "bn" and old_layer:
            if attr_name == "weight":
                layer_array[-1]['bn_weights'] = tens
            if attr_name == "running_var":
                layer_array[-1]['bn_running_var'] = tens
    check_layer_names(layer_array)
    return layer_array


def check_layer_names(layer_array):
    layer_names = [x['layer'] for x in layer_array]
    layer_name_problem = "Problem with layer names!"
    for i in range(len(layer_names)):
        for j in range(i + 1, len(layer_names)):
            assert layer_names[i] != layer_names[j], layer_name_problem


def load_model_weights_pytorch(model_path, pytorch_device):
    """
    Take a pytorch saved model state dict, and return an array of the weight
    tensors as numpy arrays
    NB: this relies on the dict being ordered in the right order.
    model_path: a string, pointing to a pytorch saved state_dict
    pytorch_device: pytorch device, which device to save the model to
    returns: an array of dicts containing layer names and various numpy
             tensors
    """
    state_dict = torch.load(model_path, map_location=pytorch_device)
    layer_array = get_weight_tensors_from_state_dict(state_dict)
    for layer_dict in layer_array:
        for key, val in layer_dict.items():
            if isinstance(val, torch.Tensor):
                layer_dict[key] = val.detach().cpu().numpy()
    return layer_array


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
