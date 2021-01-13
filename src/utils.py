import collections
import time

import numpy as np
import scipy.sparse as sparse
import torch

# potential refactor: graph_utils.py that involves doing graph theory stuff,
# utils.py for stuff like get_random_int_time


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


def get_graph_weights_from_state_dict(state_dict):
    """
    Takes a pytorch state dict, and returns an array of the weight tensors that
    constitute the graph we're working with.
    NB: relies on the dict having the expected order.
    state_dict: a pytorch state dict
    returns: an array of pytorch tensors
    """
    assert isinstance(state_dict, collections.OrderedDict)
    weights = []
    for string in state_dict:
        if string.endswith("weight"):
            weights.append(state_dict[string])
    return weights


def load_model_weights_pytorch(model_path, pytorch_device):
    """
    Take a pytorch saved model state dict, and return an array of the weight
    tensors as numpy arrays
    NB: this relies on the dict being ordered in the right order.
    model_path: a string, pointing to a pytorch saved state_dict
    pytorch_device: pytorch device, which device to save the model to
    returns: array of numpy arrays of weight tensors (no biases)
    """
    state_dict = torch.load(model_path, map_location=pytorch_device)
    torch_weights = get_graph_weights_from_state_dict(state_dict)
    np_weights = [tens.detach().cpu().numpy() for tens in torch_weights]
    return np_weights


def delete_isolated_ccs(weights_array, adj_mat):
    """
    Deletes isolated connected components from the graph - that is, connected
    components that don't have vertices in both the first and the last layers.
    weights_array: array of numpy arrays, representing weights of the NN
    adj_mat: sparse adjacency matrix of the graph.
    return a tuple: first element is an updated weights_array, second element
    is an updated adj_mat, third element is array of arrays of deleted rows,
    fourth element is array of arrays of deleted cols
    """
    # TODO: check that rows to delete and cols to delete match
    nc, labels = sparse.csgraph.connected_components(adj_mat, directed=False)
    # if there's only one connected component, don't bother
    empty_del_array = [[] for _ in weights_array]
    if nc == 1:
        return weights_array, adj_mat, empty_del_array, empty_del_array
    widths = weights_to_layer_widths(weights_array)
    cum_sums = np.cumsum(widths)
    cum_sums = np.insert(cum_sums, 0, 0)
    initial_ccs = set(labels[i] for i in range(cum_sums[0], cum_sums[1]))
    final_ccs = set(labels[i] for i in range(cum_sums[-2], cum_sums[-1]))
    isolated_ccs = set(range(nc)).difference(
        initial_ccs.intersection(final_ccs))
    # if there aren't isolated ccs, don't bother deleting any
    if not isolated_ccs:
        return weights_array, adj_mat, empty_del_array, empty_del_array
    # go through weights_array, construct new one without rows and cols in
    # isolated clusters
    new_weights_array = []
    all_deleted_rows = []
    all_deleted_cols = []
    for t, tensor in enumerate(weights_array):
        # remember: in pytorch, rows are out_{channels, neurons}, columns are
        # in_{channels, neurons}
        rows_to_delete = []
        for i in range(tensor.shape[0]):
            node = cum_sums[t + 1] + i
            if labels[node] in isolated_ccs:
                rows_to_delete.append(i)

        cols_to_delete = []
        for j in range(tensor.shape[1]):
            node = cum_sums[t] + j
            if labels[node] in isolated_ccs:
                cols_to_delete.append(j)

        rows_deleted = np.delete(tensor, rows_to_delete, 0)
        new_tensor = np.delete(rows_deleted, cols_to_delete, 1)
        new_weights_array.append(new_tensor)
        all_deleted_rows.append(rows_to_delete)
        all_deleted_cols.apppend(cols_to_delete)
    new_adj_mat = weights_to_graph(new_weights_array)
    return new_weights_array, new_adj_mat, all_deleted_rows, all_deleted_cols


def invert_deleted_neurons_np(tens, rows_deleted, cols_deleted):
    """
    Takes a numpy array `tens`, and two lists of ints.
    Returns a numpy array which, if you deleted every index of the first
    dimension in rows_deleted and every index of the second dimension in
    cols_deleted (i.e. all the rows in rows_deleted and all the cols in
    cols_deleted, for a 2d array), then you'd get the input back. The entries
    that would be deleted are input as 0.0.
    """
    if len(rows_deleted) > 0:
        assert tens.shape[0] > rows_deleted[-1]
    if len(cols_deleted) > 0:
        assert tens.shape[1] > cols_deleted[-1]
    for row in rows_deleted:
        tens = np.insert(tens, row, 0.0, axis=0)
    for col in cols_deleted:
        tens = np.insert(tens, col, 0.0, axis=1)
    return tens


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
    nones_row = [init_zeros] + [None] * len(weights_array)
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
