import copy

import numpy as np
import torch
import torch.nn as nn
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

from networks import cnn_dict, mlp_dict
from train_model import csordas_loss, eval_net, load_datasets
from utils import get_weight_tensors_from_state_dict

# can I even export eval_net? will that mess me over somehow with the _run
# thing?

# TODO: add imports as necessary
# Warning: don't apply to network while pruning is happening.

ablation_acc_test = Experiment('ablation_acc_test')
ablation_acc_test.captured_out_filter = apply_backspaces_and_linefeeds
ablation_acc_test.observers.append(FileStorageObserver('ablation_acc_runs'))


@ablation_acc_test.config
def basic_config():
    blah = 'blah'
    _ = locals()
    del _


# helper functions:
# - get mask from cluster
# - apply mask to live net (almost in utils but not quite)
# - ... run whole experiment? this seems pretty easy!

# so, I probably need the cluster as well as the deleted neurons
# spectral_cluster_model dumps ((n_cut_val, clustering_labels),
#                               isolation_indicator), so does
# shuffle_and_cluster (in a different way)
# so I can use those, and assume that I get a path to the run.json
# (let's say in dict form)
# actually, I should just edit spectral_cluster_model to return a similar dict.

# to apply masks, can just multiply tensors.

# one tricky thing: we only cluster a subset of neurons in general (for e.g.
# CNNs)

# TODOS:
# - refactor compare_masks_clusters to share functions with this file
#   (probably by adding cluster_utils file)
# - also add documentation to compare_masks_clusters
# - change how spectral_cluster_model outputs stuff


def mask_from_cluster(cluster, cluster_labels, isolation_indicator,
                      layer_widths):
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
    Returns: array of pytorch tensors, with entry 0 for those going between
        neurons in the selected cluster and 1 otherwise.
    """
    padded_labels = pad_labels(cluster_labels, isolation_indicator)
    layer_labels = split_labels(padded_labels, layer_widths)
    clust_mask = []
    for i in range(len(layer_widths)):
        this_width = layer_widths[i]
        next_width = layer_widths[i + 1]
        mask = layer_mask_from_labels_numpy(this_width, next_width,
                                            layer_labels[i],
                                            layer_labels[i + 1], cluster)
        torch_mask = torch.from_numpy(mask)
        clust_mask.append(torch_mask)
    return clust_mask


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
    Returns: an out_width by in_width numpy array, with zeros for weights
        between two neurons in the selected cluster, and ones in other
        locations.
    """
    my_mat = np.ones((out_width, in_width))
    in_labels_np = np.array(in_labels)
    out_labels_np = np.array(out_labels)
    my_mat[np.ix_(out_labels_np == cluster, in_labels_np == cluster)] = 0
    return my_mat


def masks_from_clusters(num_clusters, cluster_labels, isolation_indicator,
                        layer_widths):
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
    Returns: array of arrays of pytorch tensors. ith array is for the ith
        cluster. jth sub-array is for the jth weights layer. entry 0 for
        weights going between neurons in the selected cluster, 1 otherwise.
    """
    clust_masks = []
    for i in range(num_clusters):
        clust_masks.append(
            mask_from_cluster(i, cluster_labels, isolation_indicator,
                              layer_widths))
    return clust_masks


def apply_mask_to_net(mask_array, state_dict, net_type):
    """
    Mask out weights according to a given mask_array
    mask_array: array of pytorch tensors, which should be masks for layers of a
        network.
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
        if layer_name not in layer_names:
            if module_type in ["fc", "conv"]:
                layer_names.append((layer_name, module_type))

    def matches(net_type, module_type):
        return (net_type, module_type) in [("mlp", "fc"), ("cnn", "conv")]

    # mask out the right layers
    mask_ind = 0
    for i, layer_tup in enumerate(layer_names):
        layer_name = layer_tup[0]
        module_type = layer_tup[1]
        weight_name = layer_name + "." + module_type + ".weight"
        if matches(net_type, module_type) and i != len(layer_names) - 1:
            mask_tens = mask_array[mask_ind]
            mask_ind += 1
            weight = state_dict_copy[weight_name]
            shaped_mask_tens = mask_tens
            for i in range(mask_tens.dim, weight.dim):
                shaped_mask_tens = torch.unsqueeze(shaped_mask_tens, -1)
            state_dict_copy[weight_name] = torch.mul(shaped_mask_tens, weight)
    return state_dict_copy


# next: load stuff from saved files, get test set, find accuracy of
# ablated nets.

# what I need to get:
# - state_dict
# - layer_widths (should be able to get from state dict)
# - cluster_labels
# - isolation_indicator
# - network class (to turn state_dict into network)
# - test set
# - net_type

# tasks to perform:
# - load all those things from saved file - but how are you going to get the
#   network class...
# - use cluster_labels to get num_clusters
# - layer_widths comes from, i guess, extracting weights from state_dict using
#   net_type? seems wasteful but ok
# - make all the masks
# - for each mask: mask out the network, get test accuracy, including on
#   sub-tasks.


def magic(x):
    return x


def get_ablation_accuracies(cluster_labels, isolation_indicator, state_dict,
                            net_type, net_name, dataset, batch_size):
    num_clusters = len(set(cluster_labels))
    weights_array = get_weight_tensors_from_state_dict(state_dict)
    for layer_dict in weights_array:
        for key, val in layer_dict.items():
            if isinstance(val, torch.Tensor):
                layer_dict[key] = val.detach().cpu().numpy()

    layer_widths = magic(weights_array)
    # TODO: less magic
    mask_arrays = masks_from_clusters(num_clusters, cluster_labels,
                                      isolation_indicator, layer_widths)
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


# current TODOS: get layer width nicely, and then wrap this up into something
# that can run when given a results json
