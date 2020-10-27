import copy
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


def load_model_weights_pytorch(model_path, model_class, pytorch_device):
    """
    Take a pytorch saved model, and return an array of the weight tensors
    model_path: a string
    model_class: a pytorch class that probably has to inherit from
                 torch.nn.Module
    pytorch_device: pytorch device, which device to save the model to
    returns: array of numpy arrays of weight tensors (no biases)
    """
    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=pytorch_device))
    param_list = list(model.parameters())
    assert len(param_list) % 2 == 0
    # param_shapes = [tensor.size() for tensor in param_list]
    # print(param_shapes)
    weight_params = [
        copy.deepcopy(param_list[2 * j].detach().numpy())
        for j in range(len(param_list) // 2)
    ]
    return weight_params
