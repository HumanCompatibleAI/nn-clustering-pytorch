import copy
import json

import numpy as np
import torch
import torch.nn as nn
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

from graph_utils import np_layer_array_to_graph_weights_array
from networks import cnn_dict, mlp_dict
from train_model import csordas_loss, eval_net, load_datasets
from utils import get_weight_tensors_from_state_dict, weights_to_layer_widths

# Warning: don't apply to network while pruning is happening.

# TODOS:
# - refactor compare_masks_clusters to share functions with this file
#   (probably by adding cluster_utils file)

ablation_acc_test = Experiment('ablation_acc_test')
ablation_acc_test.captured_out_filter = apply_backspaces_and_linefeeds
ablation_acc_test.observers.append(FileStorageObserver('ablation_acc_runs'))


@ablation_acc_test.config
def basic_config():
    training_dir = './training_runs/105/'
    shuffle_cluster_dir = './shuffle_clust_runs/82/'
    pre_mask_path = None
    is_pruned = False
    _ = locals()
    del _


def mask_from_cluster(cluster, cluster_labels, isolation_indicator,
                      layer_widths, net_type):
    """
    Make a pytorch mask tensor for the weights within a cluster.
    cluster: int, representing which cluster to deal with
    cluster_labels: array of ints, entry n giving the cluster of neuron n.
        Should not yet be padded.
    isolation_indicator: array, entry n being 1 if neuron n of the network is
        isolated and 0 otherwise.
    layer_widths: array of ints, entry i being the width of layer i of the net.
        This should range only over the parts of the net that clustering is
        applied to.
    net_type: string, indicating whether the network is an mlp or a cnn
    Returns: array of pytorch tensors, with entry False for those going between
        neurons in the selected cluster and True otherwise.
    """
    # DANGER: we don't really use the layer name field, which might mean we
    # make mistakes.
    padded_labels = pad_labels(cluster_labels, isolation_indicator)
    layer_labels = split_labels(padded_labels, layer_widths)
    weights_name = net_type_to_weights_name(net_type)
    clust_mask = []
    for i in range(len(layer_widths) - 1):
        this_width = layer_widths[i]
        next_width = layer_widths[i + 1]
        mask = layer_mask_from_labels_numpy(this_width, next_width,
                                            layer_labels[i],
                                            layer_labels[i + 1], cluster)
        torch_mask = torch.from_numpy(mask)
        clust_mask.append({'layer': str(i), weights_name: torch_mask})
    return clust_mask


def net_type_to_weights_name(net_type):
    """
    Take a net_type, return the name of fields in layer dicts that will return
    us weight tensors
    net_type: string, indicating whether the network is an mlp or a cnn
    Returns: string
    """
    return 'fc_weights' if net_type == 'mlp' else 'conv_weights'


def pad_labels(cluster_labels, isolation_indicator):
    """
    Take an array of cluster labels, and insert an entry "-1" for each isolated
    node that ended up deleted from the network.
    cluster_labels: array. Entry n gives the cluster label of neuron n of the
        network, iterating from the first layer of the network to the last,
        skipping isolated neurons
    isolation_indicator: array. Entry n is 1 if neuron n is isolated, 0
        otherwise, iterating from the first layer of the network to the last.
    Returns: an array of cluster labels.
    """
    for i, val in enumerate(isolation_indicator):
        if val == 1:
            cluster_labels.insert(i, -1)
    return cluster_labels


def split_labels(cluster_labels, layer_widths):
    """
    Take an array of cluster labels, and return one array for each layer of the
    network.
    cluster_labels: array. Entry n gives the cluster label of neuron n of the
        network, where n should iterate over every neuron, starting from the
        first layer.
    layer_widths: array of the width of each relevant layer of the network.
    Returns: an array of arrays, giving cluster labels for each layer of the
        network.
    """
    layer_labels = []
    prev_width = 0
    for width in layer_widths:
        layer_labels.append(cluster_labels[prev_width:prev_width + width])
        prev_width += width
    return layer_labels


def layer_mask_from_labels_numpy(in_width, out_width, in_labels, out_labels,
                                 cluster):
    """
    Make a numpy array that can mask a weight tensor of a neural network,
    selecting weights that go from a neuron in the selected cluster,
    to another neuron in the selected cluster.
    in_width: int, width of incoming layer
    out_width: int, width of outgoing layer
    in_labels: array of ints, element i is the cluster label of neuron i of the
        input layer
    out_labels: array of ints, element i is the cluster label of neuron i of
        the output layer
    cluster: int, representing which cluster to make a mask for
    Returns: an out_width by in_width numpy array, with Falses for weights
        between two neurons in the selected cluster, and Trues in other
        locations.
    """
    my_mat = np.full((out_width, in_width), True)
    in_labels_np = np.array(in_labels)
    out_labels_np = np.array(out_labels)
    my_mat[np.ix_(out_labels_np == cluster, in_labels_np == cluster)] = False
    return my_mat


def masks_from_clusters(num_clusters, cluster_labels, isolation_indicator,
                        layer_widths, net_type):
    """
    Make a pytorch mask tensor for each cluster that masks out the weights
    within that cluster.
    num_clusters: int, representing the number of clusters
    cluster_labels: array of ints, entry n giving the cluster of neuron n.
        Should not yet be padded.
    isolation_indicator: array, entry n being 1 if neuron n of the network is
        isolated and 0 otherwise.
    layer_widths: array of ints, entry i being the width of layer i of the net.
        This should range only over the parts of the net that clustering is
        applied to.
    net_type: string, indicating whether the network is an mlp or a cnn
    Returns: array of arrays of pytorch tensors. ith array is for the ith
        cluster. jth sub-array is for the jth weights layer. entry 0 for
        weights going between neurons in the selected cluster, 1 otherwise.
    """
    clust_masks = []
    for i in range(num_clusters):
        clust_masks.append(
            mask_from_cluster(i, cluster_labels, isolation_indicator,
                              layer_widths, net_type))
    return clust_masks


def matches(net_type, module_type):
    return (net_type, module_type) in [("mlp", "fc"), ("cnn", "conv")]


def net_type_to_bias_name(net_type):
    """
    Take a net_type, return the name of fields in layer dicts that will return
    us bias tensors
    net_type: string, indicating whether the network is an mlp or a cnn
    Returns: string
    """
    return 'fc_biases' if net_type == 'mlp' else 'conv_biases'


def apply_mask_to_net(mask_array, state_dict, net_type):
    """
    Mask out weights according to a given mask_array
    mask_array: array of dicts containing pytorch tensors, which should be
        masks for layers of a network.
    state_dict: a pytorch state dict (which should be an ordered dict)
    net_type: string 'mlp' or 'cnn'
    returns: a new state_dict.
    """
    state_dict_copy = copy.deepcopy(state_dict)
    # get layer names, so that we can count layers.
    layer_names = []
    for name, tens in state_dict_copy.items():
        name_parts = name.split('.')
        assert len(name_parts) == 3, name
        layer_name = name_parts[0]
        module_type = name_parts[1]
        attr_name = name_parts[2]
        if layer_name not in map(lambda x: x[0], layer_names):
            if module_type in ["fc", "conv"] and attr_name == "weight":
                layer_names.append((layer_name, module_type))

    # mask out the right layers
    # if we were using layer names nicely, this checking code could be much
    # simpler...
    mask_ind = 0
    in_right_block = False
    num_in_block = 0
    for i, layer_tup in enumerate(layer_names):
        layer_name = layer_tup[0]
        module_type = layer_tup[1]
        weight_name = layer_name + "." + module_type + ".weight"
        bias_name = layer_name + "." + module_type + ".bias"
        if not in_right_block and matches(net_type, module_type):
            in_right_block = True
        if in_right_block:
            if not matches(net_type, module_type) or i == len(layer_names) - 1:
                break
            if net_type != "cnn" or num_in_block != 0:
                mask_weight_field_name = net_type_to_weights_name(net_type)
                mask_bias_field_name = net_type_to_bias_name(net_type)
                mask_tens = mask_array[mask_ind][mask_weight_field_name]
                mask_ind += 1
                weight = state_dict_copy[weight_name]
                shaped_mask_tens = mask_tens
                for _ in range(mask_tens.dim(), weight.dim()):
                    shaped_mask_tens = torch.unsqueeze(shaped_mask_tens, -1)
                shaped_mask_numpy = (
                    shaped_mask_tens.cpu().detach().numpy().astype(int))
                weight_numpy = weight.cpu().detach().numpy()
                state_dict_copy[weight_name] = torch.from_numpy(
                    np.multiply(weight_numpy, shaped_mask_numpy))
                if (mask_bias_field_name in mask_array[mask_ind]
                        and mask_array[mask_ind][mask_bias_field_name]
                        is not None):
                    mask_bias = mask_array[mask_ind][mask_bias_field_name]
                    mask_bias_numpy = (
                        mask_bias.cpu().detach().numpy().astype(int))
                    bias = state_dict_copy[bias_name]
                    bias_numpy = bias.cpu().detach().numpy()
                    state_dict_copy[bias_name] = torch.from_numpy(
                        np.multiply(bias_numpy, mask_bias_numpy))
                num_in_block += 1
    return state_dict_copy


def get_ablation_accuracies(cluster_labels, isolation_indicator, state_dict,
                            net_type, net_name, dataset, batch_size):
    """
    Take a state dict, ablate each cluster in that network, and check accuracy
    post-ablation on all test sets
    cluster_labels: array of ints, entry n giving the cluster of neuron n.
        Should not yet be padded.
    isolation_indicator: array, entry n being 1 if neuron n of the network is
        isolated and 0 otherwise.
    state_dict: a pytorch state dict (which should be an ordered dict)
    net_type: string 'mlp' or 'cnn'
    net_name: string specifying the network, in conjunction with net_type.
        Will be used to index the dicts in src/networks.py to get a net class.
    dataset: string specifying the dataset
    batch_size: int specifying the test set batch size
        (I wish I didn't need this but I do)
    Returns: dict. keys are ints specifying clusters. values are dicts, whose
        keys are names of test sets (strings) and whose values are tuples of
        floats: the accuracy and loss of the network with that cluster ablated
        on that test set.
    """
    num_clusters = len(set(cluster_labels))
    # get the layer widths
    weights_array = get_weight_tensors_from_state_dict(state_dict)
    for layer_dict in weights_array:
        for key, val in layer_dict.items():
            if isinstance(val, torch.Tensor):
                layer_dict[key] = val.detach().cpu().numpy()
    graph_weights = np_layer_array_to_graph_weights_array(
        weights_array, net_type)
    layer_widths = weights_to_layer_widths(graph_weights)

    # get the masks
    mask_arrays = masks_from_clusters(num_clusters, cluster_labels,
                                      isolation_indicator, layer_widths,
                                      net_type)

    # get masked accuracy stats
    net_dict = mlp_dict if net_type == 'mlp' else cnn_dict
    net_class = net_dict[net_name]
    device = (torch.device("cuda")
              if torch.cuda.is_available() else torch.device("cpu"))
    criterion = nn.CrossEntropyLoss() if dataset != 'add_mul' else csordas_loss
    cluster_stats = {}
    for i in range(num_clusters):
        mask_array = mask_arrays[i]
        new_state_dict = apply_mask_to_net(mask_array, state_dict, net_type)
        network = net_class()
        network.load_state_dict(new_state_dict)
        test_sets = load_datasets(dataset, batch_size)[1]
        loss_dict = {}
        for test_set_name in test_sets:
            test_loader = test_sets[test_set_name]
            test_acc, test_loss = eval_net(network, test_set_name, test_loader,
                                           device, criterion, dataset)
            loss_dict[test_set_name] = (test_acc, test_loss)
        cluster_stats[i] = loss_dict
    return cluster_stats


@ablation_acc_test.automain
def run_ablation_accuracy(training_dir, pre_mask_path, shuffle_cluster_dir,
                          is_pruned):
    """
    Gets all the relevant data from config/run files, and runs
    get_ablation_accuracies.
    training_dir: string specifying the sacred directory of the relevant
        training run.
    pre_mask_path: string specifying the location of a mask to pre-apply, or
        None if no mask should be pre-applied.
    shuffle_cluster_dir: string specifying the sacred directory of the
        relevant shuffle_and_cluster run.
    """
    with open(training_dir + "config.json") as f:
        training_config = json.load(f)
    with open(shuffle_cluster_dir + "run.json") as f:
        shuffle_cluster_run = json.load(f)
    cluster_labels = shuffle_cluster_run['result']['labels']
    isolation_indicator = shuffle_cluster_run['result']['isolation_indicator']
    net_type = training_config['net_type']
    dataset = training_config['dataset']
    net_name = training_config['net_choice']
    batch_size = training_config['batch_size']
    save_path_prefix = training_config['save_path_prefix']
    save_path = save_path_prefix + ('.pth' if is_pruned else '_unpruned.pth')
    state_dict = torch.load(save_path)
    if pre_mask_path is not None:
        pre_mask_dict = torch.load(pre_mask_path)
        pre_mask_array = get_weight_tensors_from_state_dict(
            pre_mask_dict, include_biases=True)
        state_dict = apply_mask_to_net(pre_mask_array, state_dict, net_type)
    cluster_stats = get_ablation_accuracies(cluster_labels,
                                            isolation_indicator, state_dict,
                                            net_type, net_name, dataset,
                                            batch_size)
    return cluster_stats
