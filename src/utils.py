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


def invert_layer_masks_np(mat, mask_rows, mask_cols):
    """
    Takes a numpy array mat, and two lists of booleans.
    Returns a numpy array which, if masked by the lists, would produce
    the input. The entries that would be masked are input as 0.0.
    """
    assert mat.shape[0] == len(list(filter(None, mask_rows)))
    assert mat.shape[1] == len(list(filter(None, mask_cols)))
    for (row, mask_bool) in enumerate(mask_rows):
        if not mask_bool:
            mat = np.insert(mat, row, 0.0, axis=0)
    for (col, mask_bool) in enumerate(mask_cols):
        if not mask_bool:
            mat = np.insert(mat, col, 0.0, axis=1)
    return mat
