# Based on code written by Shlomi Hod, Stephen Casper, and Daniel Filan

import numpy as np
import torch
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

from datasets import load_datasets
from networks import CNN_DICT, MLP_DICT
from spectral_cluster_model import layer_array_to_clustering_and_quality
from utils import (
    compute_percentile,
    get_random_int_time,
    load_activations_numpy,
    masked_weights_from_state_dicts,
    model_weights_from_state_dict_numpy,
)

shuffle_and_clust = Experiment('shuffle_and_clust')
shuffle_and_clust.captured_out_filter = apply_backspaces_and_linefeeds
shuffle_and_clust.observers.append(FileStorageObserver('shuffle_clust_runs'))

# for the moment I won't deal with simple math config, too much of a pain


@shuffle_and_clust.config
def basic_config():
    num_clusters = 4
    weights_path = "./models/mlp_kmnist.pth"
    mask_path = None
    net_type = 'mlp'
    shuffle_method = "all"
    normalize_weights = True
    epsilon = 1e-9
    num_samples = 100
    eigen_solver = 'arpack'
    dataset = None
    net_str = None
    activations_path = None
    generate_acts = False
    _ = locals()
    del _


@shuffle_and_clust.named_config
def cnn_config():
    net_type = 'cnn'
    _ = locals()
    del _


@shuffle_and_clust.named_config
def activations_config():
    dataset = 'kmnist'
    net_str = 'mnist'
    activations_path = "./models/mlp_kmnist_activations.pth"
    generate_acts = True
    _ = locals()
    del _


def shuffle_weight_tensor(weight_tensor):
    """
    Permute all the elements of an input weight tensor (not in-place).
    weight_tensor: a numpy tensor.
    returns a new tensor with permuted weights.
    """
    tensor_shape = weight_tensor.shape
    flat_tensor = weight_tensor.flatten()
    rand_flat = np.random.permutation(flat_tensor)
    return np.reshape(rand_flat, tensor_shape)


def shuffle_weight_tensor_nonzero(weight_tensor):
    """
    Permute all the non-zero elements of an input weight tensor (not in-place).
    weight_tensor: a numpy tensor.
    returns a new tensor with permuted weights.
    """
    tensor_shape = weight_tensor.shape
    flat_tensor = weight_tensor.flatten()
    nonzero_indices = np.nonzero(flat_tensor)[0]
    perm = np.random.permutation(len(nonzero_indices))
    permuted_flat_tensor = np.zeros_like(flat_tensor)
    permuted_flat_tensor[nonzero_indices] = flat_tensor[nonzero_indices[perm]]
    return np.reshape(permuted_flat_tensor, tensor_shape)


def shuffle_state_dict(shuffle_func, state_dict):
    """
    Apply shuffle_func to all tensors within a state dict
    shuffle_func: a function from numpy ndarrays to numpy ndarrays
    layer_array: a list of dicts of layer info, including some numpy ndarrays
    returns: a new layer array with permuted weights
    """
    new_state_dict = {}
    for key, val in state_dict.items():
        if isinstance(val, torch.Tensor):
            val_np = val.detach().cpu().numpy()
            shuffle_val_np = shuffle_func(val_np)
            new_val = torch.from_numpy(shuffle_val_np)
        else:
            new_val = val
        new_state_dict[key] = new_val
    return new_state_dict


def shuffle_and_cluster(num_samples, state_dict, net_type, num_clusters,
                        net_str, dataset, eigen_solver, normalize_weights,
                        epsilon, shuffle_method, seed_int):
    """
    shuffles a weights array a number of times, then finds the n-cut of each
    shuffle.
    num_samples: an int for the number of shuffles.
    state_dict: state dict of the network, which containts torch tensors.
    net_type: string indicating whether the network is an MLP or a CNN.
    num_clusters: an int for the number of clusters to cluster into.
    net_str: string indicating which network architecture is being used (in
             order to play with activations), or None if activations aren't
             being used.
    dataset: string indicating what dataset should be used to get activations,
             or None if activations aren't being used.
    eigen_solver: a string or None specifying which eigenvector solver spectral
                  clustering should use.
    normalize_weights: bool specifying whether the weights should be
                       'normalized' before clustering.
    epsilon: a small positive float for stopping us from dividing by zero.
    seed_int: an integer to set the numpy random seed to determine the
              shufflings.
    returns an array of floats
    """
    # get activations
    # then turn to layer array, etc.
    assert shuffle_method in ["all", "nonzero"]
    assert epsilon > 0
    shuffle_func = (shuffle_weight_tensor if shuffle_method == "all" else
                    shuffle_weight_tensor_nonzero)
    np.random.seed(seed_int)
    n_cuts = []
    for i in range(num_samples):
        print("shuffle ", i)
        shuffled_state_dict = shuffle_state_dict(shuffle_func, state_dict)
        shuffled_layer_array = model_weights_from_state_dict_numpy(
            shuffled_state_dict)
        if net_str is not None:
            assert dataset is not None
            acts_dict = get_activations(net_type, net_str, shuffled_state_dict,
                                        dataset)
        big_tup = layer_array_to_clustering_and_quality(
            shuffled_layer_array, net_type, acts_dict, num_clusters,
            eigen_solver, normalize_weights, epsilon)
        n_cut = big_tup[0][0]
        n_cuts.append(n_cut)
    return n_cuts


def get_activations(net_type, net_str, state_dict, dataset):
    """
    Load a dictionary of neuron activations on a specified training dataset.
    net_type (str): Whether the network is an MLP or a CNN
    net_str (str): Specification of the network architecture, to index into
        MLP_DICT or CNN_DICT.
    state_dict (dict): Dictionary of parameter tensors to load into the network
        architecture.
    dataset (str): Specification of which dataset to use.
    returns: A dictionary mapping layers to activations. 0 index varies over
        batch size, 1 index varies over neurons in a layer.
    """
    network_dict = MLP_DICT if net_type == 'mlp' else CNN_DICT
    net = network_dict[net_str]()
    net.load_state_dict(state_dict)
    train_set, _, _ = load_datasets(dataset, 128, {})
    new_batch = next(iter(train_set))
    _ = net(new_batch[0])
    acts_dict = net.activations
    for key, val in acts_dict.items():
        acts_dict[key] = val.detach().cpu().numpy()
    return acts_dict


@shuffle_and_clust.automain
def run_experiment(weights_path, mask_path, activations_path, generate_acts,
                   net_type, num_clusters, dataset, net_str, eigen_solver,
                   epsilon, num_samples, shuffle_method, normalize_weights):
    """
    load saved weights, cluster them, get their n-cut, then shuffle them and
    get the n-cut of the shuffles. Before each clustering, delete any isolated
    connected components.
    NB: if the main net has isolated connected components, those neurons will
    likely gain connections when the net is shuffled (unlike in Clusterability
    in Neural Networks arXiv:2103.03386)
    weights_path: path to where weights are saved. String suffices.
    mask_path: path to where mask is saved (as a string), or None if no mask is
               used
    activations_path: path to where activations are saved, or None if no
                      activations are used.
    generate_acts: bool indicating whether the network will be run to sample
                   activations to incorporate into initial clustering.
    net_type: string indicating whether the model is an MLP or a CNN
    num_clusters: int, number of groups to cluster the net into
    dataset: string indicating what dataset should be used to get activations,
             or None if activations aren't being used.
    net_str: string indicating which network architecture is being used (in
             order to play with activations), or None if activations aren't
             being used.
    eigen_solver: string specifying which eigenvalue solver to use for spectral
                  clustering
    epsilon: small positive number to stop us dividing by zero
    num_samples: how many shuffles to compare against
    shuffle_method: string indicating how to shuffle the network
    normalize_weights: bool specifying whether the weights should be
                       'normalized' before clustering.
    returns: dict containing 'true n-cut', a float, 'num samples', an int,
             'mean', a float representing the mean shuffled n-cut,
             'stdev', a float representing the standard deviation of the
             distribution of shuffled n-cuts,
             'percentile', a float representing the empirical percentile of the
             true n-cut among the shuffled n-cuts,
             and 'z-score', a float representing the z-score of the true n-cut
             in the distribution of the shuffled n-cuts.
    """
    device = (torch.device("cuda")
              if torch.cuda.is_available() else torch.device("cpu"))
    state_dict = torch.load(weights_path, map_location=device)
    if mask_path is not None:
        mask_dict = torch.load(mask_path, map_location=device)
        state_dict = masked_weights_from_state_dicts(state_dict, mask_dict)
    if generate_acts:
        acts_dict = get_activations(net_type, net_str, state_dict, dataset)
    elif activations_path is not None:
        acts_dict = load_activations_numpy(activations_path, device)
    else:
        acts_dict = None
    layer_array = model_weights_from_state_dict_numpy(state_dict)
    big_tup = layer_array_to_clustering_and_quality(layer_array, net_type,
                                                    acts_dict, num_clusters,
                                                    eigen_solver,
                                                    normalize_weights, epsilon)
    true_n_cut, labels = big_tup[0]
    isolation_indicator = big_tup[1]
    time_int = get_random_int_time()
    shuffled_n_cuts = shuffle_and_cluster(num_samples, state_dict, net_type,
                                          num_clusters, net_str, dataset,
                                          eigen_solver, normalize_weights,
                                          epsilon, shuffle_method, time_int)

    shuff_mean = np.mean(shuffled_n_cuts)
    shuff_stdev = np.std(shuffled_n_cuts)
    n_cut_percentile = compute_percentile(true_n_cut, shuffled_n_cuts)
    assert epsilon > 0
    z_score = (true_n_cut - shuff_mean) / (shuff_stdev + epsilon)
    result = {
        'true n-cut': true_n_cut,
        'num samples': num_samples,
        'mean': shuff_mean,
        'stdev': shuff_stdev,
        'percentile': n_cut_percentile,
        'z-score': z_score,
        'labels': labels.tolist(),
        'isolation_indicator': isolation_indicator
    }
    return result
