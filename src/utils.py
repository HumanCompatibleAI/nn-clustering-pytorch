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


def load_model_weights_pytorch(model_path, pytorch_device):
    """
    Take a pytorch saved model state dict, and return an array of the weight
    tensors
    NB: this relies on the dict being ordered in the right order.
    model_path: a string, pointing to a pytorch saved state_dict
    pytorch_device: pytorch device, which device to save the model to
    returns: array of numpy arrays of weight tensors (no biases)
    """
    state_dict = torch.load(model_path, map_location=pytorch_device)
    assert isinstance(state_dict, collections.OrderedDict)
    weights = []
    for string in state_dict:
        if string.endswith("weight"):
            weights.append(state_dict[string].detach().cpu().numpy())
    return weights
