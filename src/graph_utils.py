import itertools

import numpy as np
import scipy.sparse as sparse

from utils import (
    size_and_multiply_np,
    size_sqrt_divide_np,
    weights_to_layer_widths,
)


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
    nc, labels = sparse.csgraph.connected_components(adj_mat, directed=False)
    # if there's only one connected component, don't bother
    empty_del_array = [[] for _ in weights_array]
    empty_isolation_indicator = [0 for _ in labels]
    if nc == 1:
        return (weights_array, adj_mat, empty_del_array, empty_del_array,
                empty_isolation_indicator)
    widths = weights_to_layer_widths(weights_array)
    cum_sums = np.cumsum(widths)
    cum_sums = np.insert(cum_sums, 0, 0)
    initial_ccs = set(labels[i] for i in range(cum_sums[0], cum_sums[1]))
    final_ccs = set(labels[i] for i in range(cum_sums[-2], cum_sums[-1]))
    main_ccs = initial_ccs.intersection(final_ccs)
    isolated_ccs = set(range(nc)).difference(main_ccs)
    if not main_ccs:
        print("This neural network isn't connected from start to end.")
        return None
    # if there aren't isolated ccs, don't bother deleting any
    if not isolated_ccs:
        return (weights_array, adj_mat, empty_del_array, empty_del_array,
                empty_isolation_indicator)
    # go through weights_array, construct new one without rows and cols in
    # isolated clusters
    all_deleted_rows = []
    all_deleted_cols = []
    for t, tensor in enumerate(weights_array):
        # remember: in pytorch, rows are out_{channels, neurons}, columns are
        # in_{channels, neurons}
        if t == 0:
            inputs_to_delete = []
            for j in range(tensor.shape[1]):
                node = cum_sums[t] + j
                if labels[node] in isolated_ccs:
                    inputs_to_delete.append(j)
            all_deleted_cols.append(inputs_to_delete)
        nodes_to_delete = []
        for i in range(tensor.shape[0]):
            node = cum_sums[t + 1] + i
            if labels[node] in isolated_ccs:
                nodes_to_delete.append(i)
        all_deleted_rows.append(nodes_to_delete)
        if t != len(weights_array) - 1:
            all_deleted_cols.append(nodes_to_delete)
    rows_right_len = "all_deleted_rows has the wrong length!"
    assert len(all_deleted_rows) == len(weights_array), rows_right_len
    cols_right_len = "all_deleted_cols has the wrong length!"
    assert len(all_deleted_cols) == len(weights_array), cols_right_len
    new_weights_array = []
    for t, tensor in enumerate(weights_array):
        rows_to_delete = all_deleted_rows[t]
        cols_to_delete = all_deleted_cols[t]
        rows_deleted = np.delete(tensor, rows_to_delete, 0)
        new_tensor = np.delete(rows_deleted, cols_to_delete, 1)
        new_weights_array.append(new_tensor)
    new_adj_mat = weights_to_graph(new_weights_array)
    # get array indicating which neurons are isolated
    isolation_indicator = [1 if x in isolated_ccs else 0 for x in labels]
    return (new_weights_array, new_adj_mat, all_deleted_rows, all_deleted_cols,
            isolation_indicator)


def invert_deleted_neurons_np(tens, rows_deleted, cols_deleted):
    """
    Takes a numpy array `tens`, and two lists of ints.
    Returns a numpy array which, if you deleted every index of the first
    dimension in rows_deleted and every index of the second dimension in
    cols_deleted (i.e. all the rows in rows_deleted and all the cols in
    cols_deleted, for a 2d array), then you'd get the input back. The entries
    that would be deleted are input as 0.0.
    """
    for row in rows_deleted:
        tens = np.insert(tens, row, 0.0, axis=0)
    for col in cols_deleted:
        tens = np.insert(tens, col, 0.0, axis=1)
    return tens


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
    normalize_weights: Boolean indicating whether we should 'normalize' the
                       network before turning it into a graph, in order to
                       equate networks that differ only by the ReLU scaling
                       symmetry.
    Returns a sparse CSR matrix representing the adjacency matrix
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
        abs_mat = np.abs(mat)
        if len(abs_mat.shape) == 4:
            # take L1 norm of spatial conv filters
            abs_mat = np.sum(abs_mat, axis=(2, 3))
        sp_mat = sparse.coo_matrix(abs_mat)
        if i != len(weights_array) - 1:
            block_row = [None] * (len(weights_array) + 1)
        else:
            # add a zero matrix of the right size to the end of the last row
            # so that our final matrix is of the right size
            n = mat.shape[0]
            final_zeros = sparse.coo_matrix((n, n))
            block_row = [None] * len(weights_array)
            block_row.append(final_zeros)
        block_row[i] = sp_mat
        block_mat.append(block_row)

    # turn block_mat into a sparse matrix
    low_tri = sparse.bmat(block_mat, 'csr')
    # add to transpose to get adjacency matrix
    return low_tri + low_tri.transpose()


def np_layer_array_to_graph_weights_array(np_layer_array, net_type, eps=1e-5):
    """
    Take in a layer array of numpy tensors, and return an array of 'weight
    tensor' equivalents that can be turned into a graph. Basically, take the
    part of the network you care to turn into a graph, get the weight tensors,
    and absorb the batch norm parts.
    np_layer_array: array of dicts containing layer names and np weight tensors
    net_type: 'mlp' or 'cnn'
    eps: small positive float
    Returns: array of numpy tensors ready to be turned into a graph.
    """
    assert net_type in ['mlp', 'cnn']
    assert eps > 0
    weights_array = []
    # take the first contiguous block of layers containing desired weight
    # tensors
    weight_name = 'fc_weights' if net_type == 'mlp' else 'conv_weights'

    def has_weights(my_dict):
        return weight_name in my_dict

    for k, g in itertools.groupby(np_layer_array, has_weights):
        if k:
            weight_layers = list(g) if net_type == 'mlp' else list(g)[1:]
            break

    for layer_dict in weight_layers:
        my_weights = layer_dict[weight_name]
        if 'bn_weights' in layer_dict:
            my_weights = size_and_multiply_np(layer_dict['bn_weights'],
                                              my_weights)
        if 'bn_running_var' in layer_dict:
            my_weights = size_sqrt_divide_np(layer_dict['bn_running_var'],
                                             my_weights, eps)
        weights_array.append(my_weights)
    return weights_array


def normalize_weights_array(weights_array, eps=1e-5):
    """
    'Normalize' the weights of a network, so that for each hidden neuron, the
    norm of incoming weights to that neuron is 1. This is done by taking the
    norm x of the incoming weights, dividing all incoming weights by x, and
    then multiplying the outgoing weights of that neuron by x. For a ReLU
    network, this operation preserves network functionality.
    weights_array: array of numpy ndarrays, representing the weights of the
                   network
    eps: small positive number to ensure we never divide by 0.
    """
    new_array = []
    for x in weights_array:
        new_array.append(np.copy(x))
    for idx in range(len(new_array) - 1):
        this_layer = new_array[idx]
        next_layer = new_array[idx + 1]
        num_neurons = this_layer.shape[0]
        assert next_layer.shape[1] == num_neurons, "shapes don't match!"
        # TODO: flatten this_layer when finding norms
        this_layer_flat = this_layer.reshape(num_neurons, -1)
        scales = np.linalg.norm(this_layer_flat, axis=1)
        scales += eps
        scales_rows = np.expand_dims(scales, 1)
        for i in range(2, len(this_layer.shape)):
            scales_rows = np.expand_dims(scales_rows, i)
        scales_mul = scales
        for i in range(1, len(next_layer.shape) - 1):
            scales_mul = np.expand_dims(scales_mul, i)
        new_array[idx] = np.divide(this_layer, scales_rows)
        new_array[idx + 1] = np.multiply(next_layer, scales_mul)
    return new_array


def add_activation_gradients(weights_array, activation_dict, net_type,
                             bn_param_dicts):
    """
    Multiplies weights by the average derivative of the activation function of
    the neuron they point to. This makes the graph weights reflect partial
    derivatives of activations wrt activations, which is probably good.
    WARNING: behaviour relies on dicts being nicely ordered, which is scary.
    weights_array (array): list of numpy weight tensors of the network,
        in order.
    activation_dict (dict): dict of numpy activations of each layer, in same
        order.
    net_type (str): specifies whether network is a CNN or MLP
    bn_param_dicts (array(dict)): list of dicts of batch norm
        params for each layer, with entries empty dicts if batch norm not used
        in that layer. Params should be numpy arrays.
    returns: list of new weight tensors for a graph for the net.
    """
    def cnn_unsqueeze(tensor):
        for i in range(1, 3):
            tensor = np.expand_dims(tensor, i)
        return tensor

    num_layers = len(weights_array)
    if bn_param_dicts is not None:
        assert len(bn_param_dicts) == num_layers

    props_on = []
    for (i, act_tens) in enumerate(activation_dict.values()):
        bn_params = bn_param_dicts[i]
        if 'running_mean' in bn_params.keys():
            rmean = bn_params['running_mean']
            if net_type == 'cnn':
                rmean = cnn_unsqueeze(rmean)
            act_tens -= rmean
        if 'running_var' in bn_params.keys():
            rvar = bn_params['running_var']
            if net_type == 'cnn':
                rvar = cnn_unsqueeze(rvar)
            act_tens /= np.sqrt(rvar + 1e-5)
        if 'weight' in bn_params.keys():
            bn_weight = cnn_unsqueeze(bn_params['weight'])
            act_tens *= bn_weight
        if 'bias' in bn_params.keys():
            bn_bias = cnn_unsqueeze(bn_params['bias'])
            act_tens += bn_bias
        axes_to_collapse = 0 if net_type == 'mlp' else (0, 2, 3)
        props_on.append(
            np.mean(0.5 * (np.sign(act_tens) + 1), axis=axes_to_collapse))

    assert len(props_on) == num_layers
    new_weights = []
    for (i, (wt, prop_vec)) in enumerate(zip(weights_array, props_on)):
        if i != num_layers - 1 or net_type == 'cnn':
            for i in range(prop_vec.ndim, wt.ndim):
                prop_vec = np.expand_dims(prop_vec, i)
            new_wt = np.multiply(wt, prop_vec)
            new_weights.append(np.abs(new_wt))
        else:
            # last layer has no relu in mlps
            new_weights.append(np.abs(wt))
    return new_weights
