import itertools

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

from src.add_mul import AddMul
from src.clusterability_gradient import LaplacianEigenvalues
from src.networks import cnn_dict, mlp_dict
from src.simple_data_loader import SimpleDataLoader
from src.tiny_dataset import TinyDataset
from src.utils import (
    calc_arg_deps,
    calc_neuron_sparsity,
    get_weight_modules_from_live_net,
    size_and_multiply_np,
    size_sqrt_divide_np,
    vector_stretch,
)

train_exp = Experiment('train_model', interactive=True)
train_exp.captured_out_filter = apply_backspaces_and_linefeeds
train_exp.observers.append(FileStorageObserver('training_runs'))

# probably should define a global variable for the list of datasets.
# maybe in a utils file or something.

SIMPLE_FUNCTIONS = {
    "high_freq_waves":
    [lambda x: torch.sin(5 * x), lambda x: torch.cos(5 * x)],
    "many_fns":
    [torch.sin, lambda x: torch.log(x + 5 + 1e-3), torch.cos, torch.exp],
    "many_fns_norm": [
        lambda x: torch.sin(x) / 0.72605824,
        lambda x: torch.log(x + 5 + 1e-3) / 0.996789,
        lambda x: torch.cos(x) / 0.6598921, lambda x: torch.exp(x) / 29.747263
    ],
}


@train_exp.config
def mlp_config():
    batch_size = 128
    num_epochs = 20
    log_interval = 100
    dataset = 'kmnist'
    model_dir = './models/'
    net_type = 'mlp'
    net_choice = 'mnist'
    # pruning will be as described in Zhu and Gupta 2017, arXiv:1710.01878
    pruning_config = {
        'exponent': 3,
        'frequency': 100,
        'num_pruning_epochs': 5,
        # no pruning if num_pruning_epochs = 0
        'final_sparsity': 0.9
    }
    cluster_gradient = False
    cluster_gradient_config = {
        'num_workers': 2,
        'num_eigs': 3,
        'lambda': 1e-3,
        'frequency': 20,
        'normalize': False
    }
    optim_func = 'adam'
    optim_kwargs = {}
    decay_lr = False
    decay_lr_factor = 1
    decay_lr_epochs = 1
    training_run_string = ""
    save_path_end = (("_" + training_run_string)
                     if training_run_string != "" else "")
    save_path_prefix = (model_dir + net_type + '_' + dataset +
                        cluster_gradient * '_clust-grad' + save_path_end)

    # Simple Network config - these variables are left blank, and filled in by
    # using with simple_math_config.
    fns_name = ""
    input_type = ""
    lim = None
    num_batches_train = None
    num_batches_test = None
    calc_simple_math_diags = None
    simple_math_net_kwargs = {
        "out": None,
        "input_type": "",
        "hidden": None,
    }
    # TODO: refactor to rename this variable net_kwargs and feed it in whenever we're constructing a network

    _ = locals()
    del _


# Named config can be included by python __ with cnn_config - optional set!
@train_exp.named_config
def cnn_config():
    net_type = 'cnn'
    _ = locals()
    del _


@train_exp.named_config
def simple_math_config():
    dataset = 'simple_dataset'
    net_choice = 'simple'
    fns_name = "high_freq_waves"
    input_type = "single"
    lim = 5
    num_batches_train = 300
    num_batches_test = 10
    calc_simple_math_diags = True
    simple_math_net_kwargs = {
        "out": len(SIMPLE_FUNCTIONS[fns_name]),
        "input_type": input_type,
        "hidden": 512,
    }
    _ = locals()
    del _


# Capture means 'use sacred config to fill in variables for this function,
# unless explicitly set when called'
@train_exp.capture
def load_datasets(dataset, batch_size, fns_name=""):
    """
    get loaders for training datasets, as well as a description of the classes.
    dataset: string representing the dataset.
    batch_size: int for how many things should be in a batch
    fns_name: str, only used for simple_dataset. Gives the name of the math
        functions being modeled
    return pytorch loader for training set, pytorch loader for test set,
    tuple of names of classes.
    """

    assert dataset in [
        'mnist', 'kmnist', 'cifar10', 'tiny_dataset', 'add_mul',
        'simple_dataset'
    ]
    if dataset == 'mnist':
        return load_mnist(batch_size)
    elif dataset == 'kmnist':
        return load_kmnist(batch_size)
    elif dataset == 'cifar10':
        return load_cifar10(batch_size)
    elif dataset == 'tiny_dataset':
        return load_tiny_dataset(batch_size)
    elif dataset == 'add_mul':
        return load_add_mul(batch_size)
    elif dataset == "simple_dataset":
        fns = SIMPLE_FUNCTIONS[fns_name]
        return load_simple(fns, batch_size=batch_size)
    else:
        raise ValueError("Wrong name for dataset!")


def load_mnist(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.1307], [0.3081])])
    train_set = torchvision.datasets.MNIST(root="./datasets",
                                           train=True,
                                           download=True,
                                           transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_set = torchvision.datasets.MNIST(root="./datasets",
                                          train=False,
                                          download=True,
                                          transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=True)
    classes = ('0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine')
    return (train_loader, {'all': test_loader}, classes)


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
    return (train_loader, {'all': test_loader}, classes)


def load_cifar10(batch_size):
    normalize = transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))
    train_set = torchvision.datasets.CIFAR10(
        root="./datasets",
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(), normalize
        ]))
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_set = torchvision.datasets.CIFAR10(
        root="./datasets",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=True)
    classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse",
               "ship", "truck")
    return train_loader, {'all': test_loader}, classes


def load_tiny_dataset(batch_size):
    train_set = TinyDataset()
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=False)
    test_set = TinyDataset()
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=False)
    classes = ("0", "1")
    return train_loader, {'all': test_loader}, classes


def load_add_mul(batch_size):
    train_set = AddMul("train", 100_000, 2)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    valid_set_iid = AddMul("valid", 10_000, 2)
    iid_loader = torch.utils.data.DataLoader(valid_set_iid,
                                             batch_size=batch_size,
                                             shuffle=True)
    valid_set_add = AddMul("valid", 10_000, 2, restrict=["add"])
    add_loader = torch.utils.data.DataLoader(valid_set_add,
                                             batch_size=batch_size,
                                             shuffle=True)
    valid_set_mul = AddMul("valid", 10_000, 2, restrict=["mul"])
    mul_loader = torch.utils.data.DataLoader(valid_set_mul,
                                             batch_size=batch_size,
                                             shuffle=True)
    my_dict = {"all": iid_loader, "add": add_loader, "mul": mul_loader}
    return train_loader, my_dict, ()


@train_exp.capture
def load_simple(fns, input_type, lim, batch_size, num_batches_train,
                num_batches_test):
    train_loader = SimpleDataLoader(fns, input_type, lim, batch_size,
                                    num_batches_train)
    test_loader = SimpleDataLoader(fns, input_type, lim, batch_size,
                                   num_batches_test)
    return train_loader, {"all": test_loader}, tuple()


def csordas_get_input(data):
    """
    Converting data taken from the add_mul dataset to neural network inputs.
    Named as it is since that dataset is taken from Csordas et al 2021.
    data: a dictionary that maps strings to pytorch tensors.
    """
    onehot_inputs = F.one_hot(data["input"].long(), 10)
    onehot_op = F.one_hot(data["op"].long(), 2)
    return torch.cat([onehot_inputs.flatten(1),
                      onehot_op.flatten(1)], -1).float()


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
    norm of incoming weights to that neuron is sqrt(2), dividing the outputs
    of that neuron by the factor that the inputs were multiplied by. For a ReLU
    network, this operation preserves network functionality.
    network: a neural network. has to inherit from torch.nn.module. Currently
             probably has to be an MLP
    eps: a float that should be small relative to sqrt(2), to add stability.
    returns nothing: just modifies the network in-place
    """
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
        for i in range(2, len(inc_raw_weights.shape)):
            scales_rows = torch.unsqueeze(scales_rows, i)
        scales_mul = vector_stretch(scales, outgoing_weights.shape[1])
        for i in range(1, len(outgoing_weights.shape) - 1):
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


def get_loss(network, data, criterion, dataset, device):
    """
    Get the loss of a network on some data. This is its own function because
    the Csordas datasets work differently for this.
    network: nn.Module that's a neural network.
    data: Something you got from a dataset loader.
    criterion: a loss function.
    dataset: a string telling you what dataset you're in.
    device: pytorch device
    Returns: pytorch tensor containing a scalar.
    """
    if dataset != 'add_mul':
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = network(inputs)
        loss = criterion(outputs, labels)
    else:
        net_in = csordas_get_input(data)
        net_out = network(net_in)
        loss = criterion(net_out, data)
    return loss


@train_exp.capture
def train_and_save(network, optimizer, criterion, train_loader,
                   test_loader_dict, device, num_epochs, net_type,
                   pruning_config, cluster_gradient, cluster_gradient_config,
                   decay_lr, calc_simple_math_diags, log_interval,
                   save_path_prefix, dataset, _run):
    """
    Train a neural network, printing out log information, and saving along the
    way.
    network: an instantiated object that inherits from nn.Module or something.
    optimizer: pytorch optimizer. Might be SGD or Adam.
    criterion: loss function.
    train_loader: pytorch loader for the training dataset
    test_loader_dict: dict of pytorch loaders for the testing dataset.
                      different entries will be for different subsets of the
                      test set.
    device: pytorch device - do things go on CPU or GPU?
    num_epochs: int for the total number of epochs to train for (including any
                pruning)
    net_type: string indicating whether the net is an MLP or a CNN
    pruning_config: dict containing 'exponent', a numeric type, 'frequency',
                    int representing the number of training steps to have
                    between prunes, 'num_pruning_epochs', int representing the
                    number of epochs to prune for, and 'final_sparsity', float
                    representing how sparse the final net should be.
                    If 'num_pruning_epochs' is 0, no other elements are
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
    decay_lr: bool representing whether or not to decay the learning rate over
              training.
    calc_simple_math_diags: bool for whether to calculate the additional diags
                            for the simple network. Currently for single this
                            is just sparsity, for streamed it's sparsity and
                            argument interdependence
    log_interval: int. how many training steps between logging infodumps.
    save_path_prefix: string containing the relative path to save the model.
                      should not include final '.pth' suffix.
    dataset: string indicating the dataset
    returns: tuple of test acc, test loss, and list of tuples of
             (epoch number, iteration number, train loss)
    """
    network.to(device)
    loss_list = []

    num_pruning_epochs = pruning_config['num_pruning_epochs']
    final_sparsity = pruning_config['final_sparsity']
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
        print("\nStart of epoch", epoch)
        if decay_lr:
            decay_learning_rate(optimizer, epoch)
        network.train()
        running_loss = 0.0

        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            loss = get_loss(network, data, criterion, dataset, device)
            if (cluster_gradient
                    and i % cluster_gradient_config['frequency'] == 0):
                if cluster_gradient_config['normalize']:
                    normalize_weights(network)
                clust_reg_term = calculate_clust_reg(cluster_gradient_config,
                                                     net_type, network)
                loss += (clust_reg_term * cluster_gradient_config['frequency'])
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

        test_results_dict = {}
        for test_set in test_loader_dict:
            test_loader = test_loader_dict[test_set]
            test_acc, test_loss = eval_net(network, test_set, test_loader,
                                           device, criterion, dataset, _run)
            if dataset != "simple_dataset":
                print("Test accuracy on " + test_set + " is", test_acc)
                print("Test loss on " + test_set + " is", test_loss)
                test_results_dict[test_set] = {
                    'acc': test_acc,
                    'loss': test_loss
                }
            else:
                # Simple dataset is a regression task, don't calculate accuracy
                print("Test loss on " + test_set + " is", test_loss)
                test_results_dict[test_set] = {'loss': test_loss}
                if calc_simple_math_diags:
                    if network.input_type == "streamed":
                        # For a streamed network, calculate argument dependence
                        arg_deps = calc_arg_deps(network, device)
                        print("Argument interdependence for each stream:"
                              " {}".format(arg_deps))
                        test_results_dict[test_set]["arg_deps"] = arg_deps
                        for c, i in enumerate(arg_deps):
                            _run.log_scalar(
                                "test." + test_set + ".arg_deps"
                                ".stream_{}".format(c), i)
                    # For all simple networks, find sparsity of hidden layers
                    sparsity = calc_neuron_sparsity(network,
                                                    device,
                                                    do_print=True)
                    test_results_dict[test_set]["sparsity"] = sparsity
                    for c, i in enumerate(sparsity):
                        _run.log_scalar(
                            "test." + test_set + ".sparsity"
                            ".layer_{}".format(c + 1), i)

        if is_pruning and epoch == start_pruning_epoch - 1:
            model_path = save_path_prefix + '_unpruned.pth'
            torch.save(network.state_dict(), model_path)
            train_exp.add_artifact(model_path)
            print("Pre-pruning network saved at " + model_path)

    if is_pruning:
        for module, name in prune_params_list:
            prune.remove(module, name)
    save_path = save_path_prefix + '.pth'
    torch.save(network.state_dict(), save_path)
    train_exp.add_artifact(save_path)
    print("Network saved at " + save_path)
    return test_results_dict, loss_list


def eval_net(network, test_set, test_loader, device, criterion, dataset, _run):
    """
    gets test loss and accuracy
    network: network to get loss of
    test_set: string, name of this test set
    test_loader: pytorch loader of test set
    device: device to put data on
    criterion: loss function
    dataset: string
    returns: tuple of floats. first is test accuracy, second is test loss.
    """

    correct = 0.0
    total = 0
    loss_sum = 0.0
    num_batches = 0

    network.eval()
    with torch.no_grad():
        for data in test_loader:
            if dataset == "add_mul":
                inputs = csordas_get_input(data)
                outputs = network(inputs)
                loss = criterion(outputs, data)
                processed_outputs = process_csordas_output(outputs)
                predicted = processed_outputs.argmax(-1)
                total += outputs.shape[0]
                correct += (
                    data["output"] == predicted).all(-1).long().sum().item()
                loss_sum += loss.item()
                num_batches += 1
            else:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = network(inputs)
                _, predicted = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                total += labels.size(0)
                loss_sum += loss.item()
                num_batches += 1
                if dataset != "simple_dataset":
                    # For simple dataset, it's a regression task, so
                    # accuracy is meaningless
                    correct += (predicted == labels).sum().item()
    # For simple_dataset, test_accuracy will be 0, and should be ignored
    # It is kept in to give the function a consistent interface
    test_accuracy = correct / total
    test_avg_loss = loss_sum / num_batches
    if dataset != "simple_dataset":
        _run.log_scalar("test." + test_set + ".accuracy", test_accuracy)
    _run.log_scalar("test." + test_set + ".loss", test_avg_loss)
    return test_accuracy, test_avg_loss


@train_exp.capture
def decay_learning_rate(optimizer, epoch, decay_lr_factor, decay_lr_epochs):
    """
    Multiplies the learning rate by decay_lr_factor every decay_lr_epochs
    epochs.
    optimizer: pytorch optimizer
    epoch: int indicating the epoch number
    decay_lr_factor: float
    decay_lr_epochs: int
    """
    if epoch % decay_lr_epochs == 0 and epoch != 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay_lr_factor


def process_csordas_output(net_out):
    """
    Process the output of nets trained on Csordas datasets to have vectors for
    each digit
    """
    return net_out.view(net_out.shape[0], -1, 10)


def csordas_loss(net_out, data):
    """
    Loss function for the add_mul dataset, which has an unusual interface.
    Named as it is since that dataset is taken from Csordas et al 2021.
    """
    processed_out = process_csordas_output(net_out)
    return F.cross_entropy(processed_out.flatten(end_dim=-2),
                           data["output"].long().flatten())


@train_exp.automain
def run_training(dataset, net_type, net_choice, optim_func, optim_kwargs,
                 simple_math_net_kwargs):
    """
    Trains and saves network.
    dataset: string specifying which dataset we're using
    net_type: string indicating whether the model is an MLP or a CNN
    net_choice: string choosing which model to train
    optim_func: string specifying whether you're using adam, sgd, etc.
    optim_kwargs: dict of kwargs that you're passing to the optimizer.
    simple_math_net_kwargs: Dict of kwargs passed on to the simple math net.
        Only used if net_choice is simple
    """
    device = (torch.device("cuda")
              if torch.cuda.is_available() else torch.device("cpu"))
    if dataset == "simple_dataset":
        criterion = nn.MSELoss()
    elif dataset == "add_mul":
        criterion = csordas_loss
    else:
        criterion = nn.CrossEntropyLoss()

    if net_choice == "simple":
        network = mlp_dict[net_choice](**simple_math_net_kwargs)
    else:
        network = (mlp_dict[net_choice]()
                   if net_type == 'mlp' else cnn_dict[net_choice]())

    if optim_func == 'adam':
        optimizer_ = optim.Adam
    elif optim_func == 'sgd':
        optimizer_ = optim.SGD
    else:
        optimizer_ = optim.SGD
    optimizer = optimizer_(network.parameters(), **optim_kwargs)

    train_loader, test_loader_dict, classes = load_datasets()
    test_results_dict, loss_list = train_and_save(network, optimizer,
                                                  criterion, train_loader,
                                                  test_loader_dict, device)
    return {
        'test results dict': test_results_dict,
        'train loss list': loss_list
    }
