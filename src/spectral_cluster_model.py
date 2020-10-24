# Based on code written by Shlomi Hod, Stephen Casper, and Daniel Filan

import copy
import pickle

import numpy as np
import scipy.sparse as sparse
from pathos.multiprocessing import ProcessPool
from sklearn.cluster import SpectralClustering

from src.utils import compute_percentile, get_random_int_time

# I have no idea if that import actually works

# config variables
num_clusters = 4
weights_path = "does/not/exist.pckl"
shuffle_method = "all"

# main code


def weights_to_layer_widths(weights_array):
    """
    take in an array of weight matrices, and return how wide each layer of the
    network is
    weights_array: an array of numpy arrays representing NN layer tensors
    Returns a list of ints, each representing the width of a layer.
    """
    # TODO: add assert that shapes work the way I think they do
    layer_widths = [x.shape[1] for x in weights_array]
    layer_widths.append(weights_array[-1].shape[0])
    return layer_widths


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
    perm = np.rando.permutation(len(nonzero_indices))
    permuted_flat_tensor = np.zeros_like(flat_tensor)
    permuted_flat_tensor[nonzero_indices] = flat_tensor[nonzero_indices[perm]]
    return np.reshape(permuted_flat_tensor, tensor_shape)


def weights_to_graph(weights_array):
    """
    Take in an array of weight matrices, and return the adjacency matrix of the
    MLP that array defines.
    If the weight matrices (after taking absolute values) are A, B, C, and D,
    the adjacency matrix should be
    [[0   A^T 0   0   0  ]
     [A   0   B^T 0   0  ]
     [0   B   0   C^T 0  ]
     [0   0   C   0   D^T]
     [0   0   0   D   0  ]]
    This is because pytorch weights have index 0 for out_{features, channels}
    and index 1 for in_{features, channels}.
    weights_array: An array of 2d numpy arrays.
    Returns a sparse CSR matrix representing the adjacency matrix
    TODO: extend this to take in not just weight matrices but CNN tensors etc.
    """
    # strategy: form the lower diagonal, then transpose to get the upper diag.
    # block_mat is an array of arrays of matrices that is going to be turned
    # into a big sparse matrix in the obvious way.
    block_mat = []
    # add an initial row of Nones
    n = weights_array[0].shape[1]
    init_zeros = sparse.coo_matrix((n, n))
    nones_row = init_zeros + [None] * len(weights_array)
    block_mat.append(nones_row)

    # For everything in the weights array, add a row to block_mat of the form
    # [None, None, ..., sparsify(np.abs(mat)), None, ..., None]
    for (i, mat) in enumerate(weights_array):
        sp_mat = sparse.coo_matrix(np.abs(mat))
        if i == len(weights_array) - 1:
            # add a zero matrix of the right size to the end of the last row
            # so that our final matrix is of the right size
            n = mat.shape[0]
            final_zeros = sparse.coo_matrix((n, n))
            block_row = [None] * len(weights_array)
            block_row.append(final_zeros)
        else:
            block_row = [None] * (len(weights_array) + 1)
        block_row[i] = sp_mat
        block_mat.append(block_row)

    # turn block_mat into a sparse matrix
    low_tri = sparse.bmat(block_mat, 'csr')
    # add to transpose to get adjacency matrix
    return low_tri + low_tri.transpose()


def cluster_adj_mat(n_clusters, adj_mat, eigen_solver):
    """
    Spectrally cluster an adjacency matrix.
    n_clusters: an int representing the desired number of clusters
    adj_mat: square sparse matrix
    eigen_solver: string representing the eigenvalue solver.
                  must be one of [None, 'arpack', 'lobpcg', 'amg'].
    Returns a list of labels of length adj_mat.shape[0].
    """
    assert eigen_solver in [None, 'arpack', 'lobpcg', 'amg']
    assert adj_mat.shape[0] == adj_mat.shape[1]
    n_init = 100 if adj_mat.shape[0] > 2000 else 25
    cluster_alg = SpectralClustering(n_clusters=n_clusters,
                                     eigen_solver=eigen_solver,
                                     affinity='precomputed',
                                     assign_labels='kmeans',
                                     n_init=n_init)
    clustering = cluster_alg.fit(adj_mat)
    return clustering.labels_


def compute_n_cut(adj_mat, clustering_labels, epsilon):
    """
    Compute the n-cut of a given clustering.
    n-cut is as defined in "A Tutorial on Spectral Clustering", von Luxburg,
    2007.
    adj_mat: sparse square matrix
    clustering_labels: array of labels of length adj_mat.shape[0]
    epsilon: small positive float
    Returns a float that is the n-cut.
    """
    assert adj_mat.shape[0] == adj_mat.shape[1]
    assert epsilon > 0
    n_cut_terms = {}
    unique_labels = np.unique(
        [label for label in clustering_labels if label != -1])
    for label in unique_labels:
        out_mask = (clustering_labels != label)
        in_mask = (clustering_labels == label)
        cut = adj_mat[in_mask, :][:, out_mask].sum()
        # NB above line is slow when adj_mat is in CSR or CSC format
        vol = adj_mat[in_mask, :].sum()
        n_cut_terms[label] = 0.5 * cut / (vol + epsilon)
    return sum(n_cut_terms.values())


def adj_mat_to_cluster_quality(adj_mat, num_clusters, eigen_solver, epsilon):
    """
    Clusters a graph and computes the n-cut.
    adj_mat: sparse square adjacency matrix
    num_clusters: int representing the number of clusters to cluster into
    eigen_solver: string or None representing which eigenvalue solver to use
                  for spectral clustering
    epsilon: small positive float to stop us dividing by zero
    verbose: boolean determining whether we learn the n-cut term for each
             cluster label
    returns a float that is the n-cut value.
    """
    clustering_labels = cluster_adj_mat(num_clusters, adj_mat, eigen_solver)
    n_cut_val = compute_n_cut(adj_mat, clustering_labels, epsilon)
    return n_cut_val


def delete_isolated_ccs(weights_array, adj_mat):
    """
    Deletes isolated connected components from the graph - that is, connected
    components that don't have vertices in both the first and the last layers.
    weights_array: array of numpy arrays, representing weights of the NN
    adj_mat: adjacency matrix of the graph.
    return a tuple: first element is an updated weights_array, second element
    is an updated adj_mat.
    """
    nc, labels = sparse.csgraph.connected_components(adj_mat, directed=False)
    # if there's only one connected component, don't bother
    if nc == 1:
        return weights_array, adj_mat
    widths = weights_to_layer_widths(weights_array)
    cum_sums = np.cumsum(widths)
    cum_sums = np.insert(cum_sums, 0, 0)
    initial_ccs = set([labels[i] for i in range(cum_sums[0], cum_sums[1])])
    final_ccs = set([labels[i] for i in range(cum_sums[-2], cum_sums[-1])])
    isolated_ccs = set(range(nc)).difference(
        initial_ccs.intersection(final_ccs))
    # if there aren't isolated ccs, don't bother deleting any
    if not isolated_ccs:
        return weights_array, adj_mat
    # go through weights_array, construct new one without rows and cols in
    # isolated clusters
    new_weights_array = []
    for t, tensor in enumerate(weights_array):
        rows_to_delete = []
        for i in range(tensor.shape[0]):
            node = cum_sums[t] + i
            if labels[node] in isolated_ccs:
                rows_to_delete.append(i)

        cols_to_delete = []
        for j in range(tensor.shape[1]):
            node = cum_sums[t + 1] + j
            if labels[node] in isolated_ccs:
                cols_to_delete.append(j)

        rows_deleted = np.delete(tensor, rows_to_delete, 0)
        new_tensor = np.delete(rows_deleted, cols_to_delete, 1)
        new_weights_array.append(new_tensor)
    new_adj_mat = weights_to_graph(new_weights_array)
    return new_weights_array, new_adj_mat


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
        shuffled_weights_array_ = list(map(shuffle_func, weights_array))
        shuffled_adj_mat_ = weights_to_graph(shuffled_weights_array_)
        my_tup = delete_isolated_ccs(shuffled_weights_array_,
                                     shuffled_adj_mat_)
        shuffled_weights_array, shuffled_adj_mat = my_tup
        n_cut = adj_mat_to_cluster_quality(shuffled_adj_mat, num_clusters,
                                           eigen_solver, epsilon)
        n_cuts.append(n_cut)
    return n_cuts


def run_experiment(weights_path, num_clusters, eigen_solver, epsilon,
                   num_samples, num_workers, shuffle_method):
    """
    TODO document
    """
    with open(weights_path, 'rb') as f:
        weights_array_ = pickle.load(f)
    adj_mat_ = weights_to_graph(weights_array_)
    weights_array, adj_mat = delete_isolated_ccs(weights_array_, adj_mat_)
    true_n_cut = adj_mat_to_cluster_quality(adj_mat, num_clusters,
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
