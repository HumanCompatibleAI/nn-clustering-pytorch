import hashlib
import time

import matplotlib.pyplot as plt
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
            for module in layer_mod.children():
                if isinstance(module, torch.nn.Linear):
                    layer_dict['fc_mod'] = module
                if isinstance(module, torch.nn.Conv2d):
                    layer_dict['conv_mod'] = module
                if (isinstance(module, torch.nn.BatchNorm2d)
                        or isinstance(module, torch.nn.BatchNorm1d)):
                    layer_dict['bn_mod'] = module
            if len(layer_dict) > 1:
                layer_array.append(layer_dict)
        elif isinstance(layer_mod, torch.nn.Sequential):
            sequential_array = []
            for mod_name, module in layer_mod.named_children():
                if (isinstance(module, torch.nn.Linear)
                        or isinstance(module, torch.nn.Conv2d)):
                    name_split = mod_name.split('_')
                    my_name = layer_name + name_split[0]
                    layer_dict = {'layer': my_name}
                    if isinstance(module, torch.nn.Linear):
                        layer_dict['fc_mod'] = module
                    else:
                        layer_dict['conv_mod'] = module
                    sequential_array.append(layer_dict)
                if (isinstance(module, torch.nn.BatchNorm2d)
                        or isinstance(module, torch.nn.BatchNorm1d)):
                    sequential_array[-1]['bn_mod'] = module
            layer_array += sequential_array
    check_layer_names(layer_array)
    return layer_array


def get_weight_tensors_from_state_dict(state_dict, include_biases=False):
    """
    Takes a pytorch state dict, and returns an array of the weight tensors that
    constitute the graph we're working with.
    NB: relies on the dict having the expected order.
    NB: also relies on the network not being actively pruned
    NB: also relies on the network having the expected names and structure: see
        README for details
    NB: also relies on MLPs not using batch norm
    state_dict: a pytorch state dict
    include_biases: bool indicating whether we should include bias tensors.
                    should only be True for debugging.
    returns: an array of dicts containing layer names and various pytorch
             tensors
    """
    layer_array = []
    for name, tens in state_dict.items():
        # these names look like layer4.fc.weight or layer1.bn.running_var
        name_parts = name.split('.')
        assert len(name_parts) == 3, name
        new_layer_name = name_parts[0]
        module_name = name_parts[1]
        attr_name = name_parts[2]
        if '_' in module_name:
            module_name_parts = module_name.split('_')
            new_layer_name += module_name_parts[0]
            module_name = module_name_parts[-1]
        old_layer = (bool(layer_array)
                     and layer_array[-1]['layer'] == new_layer_name)
        if module_name == "fc" and attr_name == "weight":
            if not old_layer:
                layer_array.append({
                    'layer': new_layer_name,
                    'fc_weights': tens
                })
            if old_layer and layer_array[-1].keys() == {'layer', 'fc_biases'}:
                layer_array[-1]['fc_weights'] = tens
        if module_name == "fc" and attr_name == "bias" and include_biases:
            if old_layer:
                layer_array[-1]['fc_biases'] = tens
            else:
                layer_array.append({
                    'layer': new_layer_name,
                    'fc_biases': tens
                })
        if module_name == "conv" and attr_name == "weight":
            if not old_layer:
                layer_array.append({
                    'layer': new_layer_name,
                    'conv_weights': tens
                })
            if old_layer and layer_array[-1].keys() == {
                    'layer', 'conv_biases'
            }:
                layer_array[-1]['conv_weights'] = tens
        if module_name == "conv" and attr_name == "bias" and include_biases:
            if old_layer:
                layer_array[-1]['conv_biases'] = tens
            else:
                layer_array.append({
                    'layer': new_layer_name,
                    'conv_biases': tens
                })
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


def load_activations_numpy(activations_path, pytorch_device):
    acts_dict = torch.load(activations_path, map_location=pytorch_device)

    for key, val in acts_dict.items():
        acts_dict[key] = val.detach().cpu().numpy()

    return acts_dict


def load_model_weights_numpy(model_path, pytorch_device, include_biases=False):
    """
    Take a pytorch saved model state dict, and return an array of the weight
    tensors as numpy arrays
    NB: this relies on the dict being ordered in the right order.
    model_path: a string, pointing to a pytorch saved state_dict
    pytorch_device: pytorch device, which device to save the model to
    include_biases: bool indicating whether to also load bias tensors. should
                    only be True while debugging.
    returns: an array of dicts containing layer names and various numpy
             tensors
    """
    state_dict = torch.load(model_path, map_location=pytorch_device)
    layer_array = get_weight_tensors_from_state_dict(state_dict,
                                                     include_biases)
    for layer_dict in layer_array:
        for key, val in layer_dict.items():
            if isinstance(val, torch.Tensor):
                layer_dict[key] = val.detach().cpu().numpy()
    return layer_array


def load_masked_weights_numpy(model_path,
                              mask_path,
                              pytorch_device,
                              include_biases=False):
    """
    Load model weights as well as masks for tensors in your model, and return
    numpy ndarrays containing weights with masks applied.
    Here, 'apply' means that zeros are placed where the mask value is False
    model_path: string
    mask_path: string to state_dict of Nones and boolean tensors.
    pytorch_device: pytorch device
    include_biases: bool indicating whether to also load bias tensors. should
                    only be True while debugging.
    returns: an array of dicts containing layer names and various numpy
             tensors
    """
    model_layer_array = load_model_weights_numpy(model_path, pytorch_device,
                                                 include_biases)
    mask_layer_array = load_model_weights_numpy(mask_path, pytorch_device,
                                                include_biases)
    assert len(model_layer_array) == len(mask_layer_array)
    new_layer_array = []
    for i in range(len(model_layer_array)):
        model_dict = model_layer_array[i]
        layer_name = model_dict['layer']
        mask_dict = mask_layer_array[i]
        new_dict = {'layer': layer_name}
        for key in model_dict:
            my_tens = model_dict[key]
            corresp_mask = mask_dict[key]
            shaped_mask = corresp_mask.astype(int)
            for _ in range(shaped_mask.ndim, my_tens.ndim):
                shaped_mask = np.expand_dims(shaped_mask, -1)
            if corresp_mask is not None and key != 'layer':
                my_tens = np.multiply(my_tens, shaped_mask)
                # np.place(my_tens, np.logical_not(corresp_mask), [0])
            if key != 'layer':
                new_dict[key] = my_tens
        new_layer_array.append(new_dict)
    return new_layer_array


def load_masked_out_weights_numpy(model_path,
                                  mask_path,
                                  pytorch_device,
                                  include_biases=False):
    """
    Load model weights as well as masks for tensors in your model, and return
    numpy ndarrays containing weights that would be hidden by the mask.
    model_path: string
    mask_path: string to state_dict of Nones and boolean tensors.
    pytorch_device: pytorch device
    include_biases: bool indicating whether to also load bias tensors. should
                    only be True while debugging.
    returns: an array of dicts containing layer names and various numpy
             tensors
    """
    model_layer_array = load_model_weights_numpy(model_path, pytorch_device,
                                                 include_biases)
    mask_layer_array = load_model_weights_numpy(mask_path, pytorch_device,
                                                include_biases)
    assert len(model_layer_array) == len(mask_layer_array)
    new_layer_array = []
    for i in range(len(model_layer_array)):
        model_dict = model_layer_array[i]
        layer_name = model_dict['layer']
        mask_dict = mask_layer_array[i]
        new_dict = {'layer': layer_name}
        for key in model_dict:
            my_tens = model_dict[key]
            corresp_mask = mask_dict[key]
            if corresp_mask is not None and key != 'layer':
                np.place(my_tens, corresp_mask, [0])
            if key != 'layer':
                new_dict[key] = my_tens
        new_layer_array.append(new_dict)
    return new_layer_array


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


def tensor_size_np(tensor, comp_tensor):
    """
    Expand size of tensor until it has the same number of dims as comp_tensor
    tensor: numpy tensor
    comp_tensor: numpy tensor
    Returns: numpy tensor
    """
    big_tensor = tensor
    for i in range(big_tensor.ndim, comp_tensor.ndim):
        big_tensor = np.expand_dims(big_tensor, i)
    return big_tensor


def size_and_multiply_np(wrong_size_tensor, right_size_tensor):
    """
    Expand size of wrong_size_tensor until it has the same number of dims as
    right_size_tensor, then multiply it with right_size_tensor
    inputs and outputs all numpy tensors
    """
    return np.multiply(right_size_tensor,
                       tensor_size_np(wrong_size_tensor, right_size_tensor))


def size_sqrt_divide_np(wrong_size_tensor, right_size_tensor, eps=1e-5):
    """
    Read the code.
    wrong_size_tensor, right_size_tensor: numpy tensors
    eps: small float > 0
    """
    big_tens = tensor_size_np(wrong_size_tensor, right_size_tensor)
    div_by = np.sqrt(big_tens + eps)
    return np.divide(right_size_tensor, div_by)


def daniel_hash(string):
    """
    Produce the sha256 hash of a string, then get the last 32 bits. Like
    python's hash but deterministic.
    Returns: an int.
    """
    hex_hash = hashlib.sha256(str.encode(string)).hexdigest()
    return int(hex_hash[-8:], 16)


def print_network_weights(net):
    """
    Print out the weights of a network. Probably only use when the
    network is small.
    net: Pytorch neural network, inherits from nn.Module
    """
    for param_tensor in net.state_dict():
        print(param_tensor, "\t", net.state_dict()[param_tensor])


def split_digits(n_digits, nums):
    """
    Split each integer in a numpy array of integers into an array containing
    each digit of the original integer in a separate entry, in reverse order.
    So, for example, [10, 21, 3] -> [[0,1], [1,2], [3,0]].
    Stolen from Robert Csordas' code:
    https://github.com/RobertCsordas/modules/blob/
    3c9422bfd841a1a6aa3dd5e538f5a966b820df4f/dataset/helpers/split_digits.py

    n_digits: int, number of digits of the largest integer
    nums: np.ndarray, array of integers to split
    returns: np.ndarray containing 8-bit integers
    """
    digits = []
    for d in range(n_digits):
        digits.append(nums % 10)
        nums = nums // 10

    return np.stack(digits, -1).astype(np.uint8)


def calc_neuron_sparsity(net, device, n=10000, lim=5, do_print=True):
    """
    Diagnostic function. Runs the network on n random data points, and
    returns the proportion of neurons in each layer that are never
    activated. Requires the network to have pre-activation hooks, currently
    only implemented for simple math network
    """
    if net.input_type == "streamed":
        x = (torch.rand((n, net.out)) - 0.5) * 10
    else:
        x = (torch.rand((n, 1)) - 0.5) * 10
    x = x.to(device)
    _ = net(x)
    output = []
    for i in range(1, 4):
        activated_neurons = (net.post_activation[i] != 0).sum(axis=0)
        num_neurons = (activated_neurons > 0).sum().detach().cpu().numpy()
        total_neurons = len(activated_neurons)
        if do_print:
            print("Layer {} has {} activated neurons out of {}".format(
                i, num_neurons, total_neurons))
        output.append(num_neurons / total_neurons)
    return output


def calc_arg_deps(net, device, n=100, show_plot=False, fns=None):
    """
    Diagnostic function for a streamed simple math network, calculates the
    dependence of each stream on the inputs for other streams.
    Fixes a value for all stream but one, inputs evenly spaced values for
    the remaining stream, and runs the network. Repeats for different
    values of this fixed value and takes the variance for each value in the
    main stream. Returns the average variance for each stream.
    """
    totals = []
    for pos in range(net.out):
        n = 100
        size = 5
        if show_plot:
            plt.figure(figsize=(20, 10))
        outputs = []
        for fix in np.linspace(-size, size, 21):
            fix = np.float32(fix)
            x = torch.linspace(-size, size, n)
            x_list = []
            for i in range(net.out):
                if i == pos:
                    x_list.append(x)
                else:
                    x_list.append(torch.tensor([fix for i in range(n)]))
            x_stack = torch.stack(x_list, axis=1)
            x_stack = x_stack.to(device)
            y = net(x_stack)[:, pos]
            output = y.detach().cpu().numpy()
            outputs.append(output)
            if show_plot:
                label = fns[pos](x).detach().cpu().numpy()
                plt.plot(x.detach().cpu().numpy(),
                         output - label,
                         label="network diff {}".format(fix))
        if show_plot:
            plt.legend(bbox_to_anchor=(1.05, 1))
            plt.show()
        outputs = np.stack(outputs)
        totals.append(np.mean(np.var(outputs, axis=0)))
    return totals
