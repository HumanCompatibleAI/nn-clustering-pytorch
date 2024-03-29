# Based on code written by Shlomi Hod, Stephen Casper, and Daniel Filan

import numpy as np
import torch
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

from spectral_cluster_model import layer_array_to_clustering_and_quality
from utils import (
    compute_percentile,
    get_random_int_time,
    load_masked_weights_numpy,
    load_model_weights_numpy,
)

shuffle_and_clust = Experiment('shuffle_and_clust')
shuffle_and_clust.captured_out_filter = apply_backspaces_and_linefeeds
shuffle_and_clust.observers.append(FileStorageObserver('shuffle_clust_runs'))


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
    _ = locals()
    del _


@shuffle_and_clust.named_config
def cnn_config():
    net_type = 'cnn'
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


def shuffle_layer_array(shuffle_func, layer_array):
    """
    Apply shuffle_func to all numpy arrays within a layer array
    shuffle_func: a function from numpy ndarrays to numpy ndarrays
    layer_array: a list of dicts of layer info, including some numpy ndarrays
    returns: a new layer array with permuted weights
    """
    new_layer_array = []
    for layer_dict in layer_array:
        new_layer_dict = {}
        assert isinstance(layer_dict, dict)
        for key, val in layer_dict.items():
            if isinstance(val, np.ndarray):
                new_layer_dict[key] = shuffle_func(val)
        new_layer_array.append(new_layer_dict)
    return new_layer_array


def shuffle_and_cluster(num_samples, layer_array, net_type, num_clusters,
                        eigen_solver, normalize_weights, epsilon,
                        shuffle_method, seed_int):
    """
    shuffles a weights array a number of times, then finds the n-cut of each
    shuffle.
    num_samples: an int for the number of shuffles.
    layer_array: an array of layer dicts containing numpy ndarrays.
    net_type: string indicating whether the network is an MLP or a CNN.
    num_clusters: an int for the number of clusters to cluster into.
    eigen_solver: a string or None specifying which eigenvector solver spectral
                  clustering should use.
    normalize_weights: bool specifying whether the weights should be
                       'normalized' before clustering.
    epsilon: a small positive float for stopping us from dividing by zero.
    seed_int: an integer to set the numpy random seed to determine the
              shufflings.
    returns an array of floats
    """
    assert shuffle_method in ["all", "nonzero"]
    assert epsilon > 0
    shuffle_func = (shuffle_weight_tensor if shuffle_method == "all" else
                    shuffle_weight_tensor_nonzero)
    np.random.seed(seed_int)
    n_cuts = []
    for i in range(num_samples):
        print("shuffle ", i)
        shuffled_layer_array = shuffle_layer_array(shuffle_func, layer_array)
        big_tup = layer_array_to_clustering_and_quality(
            shuffled_layer_array, net_type, num_clusters, eigen_solver,
            normalize_weights, epsilon)
        n_cut = big_tup[0][0]
        n_cuts.append(n_cut)
    return n_cuts


@shuffle_and_clust.automain
def run_experiment(weights_path, mask_path, net_type, num_clusters,
                   eigen_solver, epsilon, num_samples, shuffle_method,
                   normalize_weights):
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
    net_type: string indicating whether the model is an MLP or a CNN
    num_clusters: int, number of groups to cluster the net into
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
    layer_array = (load_model_weights_numpy(weights_path, device)
                   if mask_path is None else load_masked_weights_numpy(
                       weights_path, mask_path, device))
    big_tup = layer_array_to_clustering_and_quality(layer_array, net_type,
                                                    num_clusters, eigen_solver,
                                                    normalize_weights, epsilon)
    true_n_cut, labels = big_tup[0]
    isolation_indicator = big_tup[1]
    time_int = get_random_int_time()
    shuffled_n_cuts = shuffle_and_cluster(num_samples, layer_array, net_type,
                                          num_clusters, eigen_solver,
                                          normalize_weights, epsilon,
                                          shuffle_method, time_int)

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
