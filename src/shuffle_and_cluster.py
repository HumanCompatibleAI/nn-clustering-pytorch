# Based on code written by Shlomi Hod, Stephen Casper, and Daniel Filan

import copy

import numpy as np
import torch
from pathos.multiprocessing import ProcessPool
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

from graph_utils import delete_isolated_ccs, weights_to_graph
from spectral_cluster_model import (
    adj_mat_to_clustering_and_quality,
    weights_array_to_clustering_and_quality,
)
from utils import (
    compute_percentile,
    get_random_int_time,
    load_model_weights_pytorch,
)

shuffle_and_clust = Experiment('shuffle_and_clust')
shuffle_and_clust.captured_out_filter = apply_backspaces_and_linefeeds
shuffle_and_clust.observers.append(FileStorageObserver('shuffle_clust_runs'))


@shuffle_and_clust.config
def basic_config():
    num_clusters = 4
    weights_path = "./models/mlp_kmnist.pth"
    shuffle_method = "all"
    epsilon = 1e-9
    num_samples = 10
    num_workers = 1
    eigen_solver = 'arpack'
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


def shuffle_and_cluster(num_samples, weights_array, num_clusters, eigen_solver,
                        epsilon, shuffle_method, seed_int):
    """
    shuffles a weights array a number of times, then finds the n-cut of each
    shuffle.
    num_samples: an int for the number of shuffles
    weights_array: an array of weight tensors (numpy arrays)
    num_clusters: an int for the number of clusters to cluster into
    eigen_solver: a string or None specifying which eigenvector solver spectral
                  clustering should use
    epsilon: a small positive float for stopping us from dividing by zero
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
    for _ in range(num_samples):
        shuffled_weights_array = list(map(shuffle_func, weights_array))
        n_cut, _ = weights_array_to_clustering_and_quality(
            shuffled_weights_array, num_clusters, eigen_solver, epsilon)
        n_cuts.append(n_cut)
    return n_cuts


@shuffle_and_clust.automain
def run_experiment(weights_path, num_clusters, eigen_solver, epsilon,
                   num_samples, num_workers, shuffle_method):
    """
    load saved weights, cluster them, get their n-cut, then shuffle them and
    get the n-cut of the shuffles. Before each clustering, delete any isolated
    connected components.
    weights_path: path to where weights are saved. String suffices.
    num_clusters: int, number of groups to cluster the net into
    eigen_solver: string specifying which eigenvalue solver to use for spectral
                  clustering
    epsilon: small positive number to stop us dividing by zero
    num_samples: how many shuffles to compare against
    num_workers: how many CPUs to compute shuffle n-cuts on
    shuffle_method: string indicating how to shuffle the network
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
    weights_array_ = load_model_weights_pytorch(weights_path, device)
    adj_mat_ = weights_to_graph(weights_array_)
    weights_array, adj_mat, _, _ = delete_isolated_ccs(weights_array_,
                                                       adj_mat_)
    true_n_cut, _ = adj_mat_to_clustering_and_quality(adj_mat, num_clusters,
                                                      eigen_solver, epsilon)

    samples_per_worker = num_samples // num_workers
    shuffle_cluster_arg_det = (samples_per_worker, weights_array, num_clusters,
                               eigen_solver, epsilon, shuffle_method)
    time_int = get_random_int_time()
    if num_workers == 1:
        args = shuffle_cluster_arg_det + (time_int, )
        shuffled_n_cuts = shuffle_and_cluster(*args)
    else:
        worker_det_args = [[copy.deepcopy(arg) for _ in range(num_workers)]
                           for arg in shuffle_cluster_arg_det]
        seed_args = [time_int + i for i in range(num_workers)]
        worker_args = worker_det_args + [seed_args]
        with ProcessPool(nodes=num_workers) as p:
            parallel_shuffled_n_cuts = p.map(shuffle_and_cluster, *worker_args)
        shuffled_n_cuts = np.concatenate(parallel_shuffled_n_cuts)

    num_samples = len(shuffled_n_cuts)
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
        'z-score': z_score
    }
    return result
