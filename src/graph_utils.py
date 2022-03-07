import itertools
import math

import numpy as np
import scipy.sparse as sparse
import torch
from torch.autograd import Function

from utils import (
    size_and_multiply_np,
    size_sqrt_divide_np,
    weights_to_layer_widths,
)

# TODO March: re-check clust wrapper math, try to get around dividing by 0
# (by using maths and logic?)


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


def add_activation_gradients_np(weights_array, activation_dict, net_type,
                                bn_param_dicts):
    """
    Multiplies weights by the average derivative of the activation function of
    the neuron they point to. This makes the graph weights reflect partial
    derivatives of activations wrt activations, which is probably good.
    WARNING: behaviour relies on dicts being nicely ordered, which is scary.
    weights_array (array): list of numpy weight tensors of the network,
        in order. Code relies on this being the output of
        np_layer_array_to_graph_weights_array
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
        if i != 0:
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


def sensitivity_combine_means_sds(m1, m2, s1, s2):
    s = s1 * s2 / torch.sqrt(s1**2 + s2**2)
    m = (s**2) * ((m1 / s1**2) + (m2 / s2**2))
    return m, s


def get_front_constant(m1, m2, s1, s2, mc, sc):
    exponand_ = 0.5 * ((mc**2 / sc**2) - (m1**2 / s1**2) - (m2**2 / s2**2))
    exponand = torch.minimum(exponand_,
                             torch.tensor(30))  # to avoid getting infs
    result = (sc / (np.sqrt(2 * math.pi) * s1 * s2)) * torch.exp(exponand)
    return result


def pos_neg_factors(mc, sc):
    exp_term = sc * torch.exp(-mc**2 / (2 * sc**2)) / np.sqrt(2 * math.pi)
    erf_term = 0.5 * mc * torch.erf(mc / (np.sqrt(2) * sc))
    return ((mc / 2) + exp_term + erf_term, (mc / 2) - exp_term - erf_term)


def zero_weight_deriv(zi_mean, zi_std, zj_minus_i_mean, zj_minus_i_std):
    pre_factor = (torch.exp(-zj_minus_i_mean**2 / (2 * zj_minus_i_std**2)) /
                  (np.sqrt(2 * math.pi) * zj_minus_i_std))
    term_1 = (zi_std * torch.exp(-zi_mean**2 / (2 * zi_std**2)) /
              np.sqrt(2 * math.pi))
    term_2 = 0.5 * zi_mean * torch.erf(zi_mean / (np.sqrt(2) * zi_std))
    return pre_factor * (term_1 + term_2)


class MakeSensitivityGraph(Function):
    """
    Takes a network, and returns new 'weights' that represent the average
    partial derivative of each activation with respect to each previous layer
    activation.
    Currently assuming the network is an MLP, implement CNNs later.
    Also currently ignoring that batchnorm exists.
    """
    @staticmethod
    def forward(ctx, activation_array, *args):
        # Basically: you run module_array_to_clust_grad_input
        # apply this func to what's in that
        weight_array = [wt for wt in args]
        num_weights = len(weight_array)
        num_activations = len(activation_array)
        props_on = [
            torch.mean(0.5 * (torch.sign(act_tens) - 1), dim=0)
            for act_tens in activation_array
        ]
        sensitivities = []
        for (i, wt_tens) in enumerate(weight_array):
            if i != len(weight_array) - 1:
                prop_vec = props_on[i + 1]  # skip the input activations
                for j in range(prop_vec.ndim, wt_tens.ndim):
                    prop_vec = torch.unsqueeze(prop_vec, j)
                new_wt = torch.mul(wt_tens, prop_vec)
                sensitivities.append(torch.abs(new_wt))
            else:
                sensitivities.append(torch.abs(wt_tens))
        ctx.save_for_backward(torch.tensor(num_weights),
                              torch.tensor(num_activations), *weight_array,
                              *activation_array)
        return tuple(sensitivities)

    @staticmethod
    def backward(ctx, *dys):
        device = (torch.device("cuda")
                  if torch.cuda.is_available() else torch.device("cpu"))
        dy = [grad for grad in dys]
        (num_weights_tens, num_activations_tens,
         *misc_stuff) = ctx.saved_tensors
        num_weights = num_weights_tens.item()
        num_activations = num_activations_tens.item()
        weight_array = misc_stuff[:num_weights]
        activation_array = misc_stuff[num_weights:(num_activations +
                                                   num_weights)]
        # step 1: for every neuron, get mean + sd of that neuron's pre-relu
        # activation
        zi_means = []
        zi_stds = []
        for act_arr in activation_array:
            zi_means.append(torch.mean(act_arr, dim=0))
            zi_stds.append(torch.std(act_arr, dim=0))

        d_frac_on_d_ws = []

        for k, act_arr in enumerate(activation_array[1:], start=1):
            wt = weight_array[k - 1]
            # zi_means[k-1] has shape [n_in]
            # zi_means[k] has shape [n_out]
            # wt has shape [n_out, n_in]
            # all means, stds, etc. will have same shape as wt
            n_out, n_in = tuple(wt.shape)
            zi_mean = zi_means[k - 1]
            zi_std = zi_stds[k - 1]
            wij_zi_mean = torch.mul(wt, zi_mean)
            wij_zi_std = torch.abs(torch.mul(wt, zi_std))
            zj_means = torch.unsqueeze(zi_means[k], 1)
            zj_minus_i_mean = torch.mul(wt, zi_mean) - zj_means
            prev_acts = activation_array[k - 1]
            # prev_acts has shape [num_samples, n_in]
            # act_arr has shape [num_samples, n_out]
            expanded_acts = torch.unsqueeze(act_arr, 2)
            expanded_prev_acts = torch.unsqueeze(prev_acts, 1)
            wij_zi_samples = torch.mul(wt, expanded_prev_acts)
            zj_minus_i_std = torch.std(expanded_acts - wij_zi_samples, dim=0)
            # pretty sure this is correct
            mu_comb, sigma_comb = sensitivity_combine_means_sds(
                wij_zi_mean, wij_zi_std, zj_minus_i_mean, zj_minus_i_std)
            c = get_front_constant(wij_zi_mean, wij_zi_std, zj_minus_i_mean,
                                   zj_minus_i_std, mu_comb, sigma_comb)
            wt_pos, wt_neg = pos_neg_factors(mu_comb, sigma_comb)
            non_zero_deriv = (c / torch.abs(wt)) * torch.where(
                wt > 0, wt_pos, wt_neg)
            zero_deriv = zero_weight_deriv(zi_mean, zi_std, zj_minus_i_mean,
                                           zj_minus_i_std)
            d_frac_on_d_w = torch.where(wt == 0, zero_deriv, non_zero_deriv)
            # TODO: for small W, use a better approximation
            d_frac_on_d_ws.append(d_frac_on_d_w)

        props_on = [
            torch.mean(0.5 * (torch.sign(act_tens) - 1), dim=0)
            for act_tens in activation_array
        ]
        new_grads = []
        for (i, wt_i) in enumerate(weight_array):
            if i != len(weight_array) - 1:
                p_on = props_on[i + 1]
                # p_on has shape [n_out]
                d_frac_on_d_w = d_frac_on_d_ws[i]
                # d_frac_on_d_w has shape [n_out, n_in]
                expanded_p_on = torch.unsqueeze(p_on, 1)
                term_1 = torch.mul(dy[i], expanded_p_on)
                grad_w_contract = torch.sum(torch.mul(dy[i], wt_i),
                                            dim=1,
                                            keepdim=True)
                term_2 = torch.mul(grad_w_contract, d_frac_on_d_w)
                new_grad = (term_1 + term_2).to(device)
                # expanded_wt_i = torch.unsqueeze(wt_i, 1) # .to(device)
                # expanded_deriv = torch.unsqueeze(d_frac_on_d_w, 2)
                #                  #.to(device)
                # expanded_dy = torch.unsqueeze(dy[i], 1)# .to(device)
                # new_grad = torch.sum(expanded_dy *
                #                      (expanded_p_on * expanded_id +
                #                       expanded_wt_i * expanded_deriv),
                #                      dim=2)
                new_grads.append(new_grad)
            else:
                new_grads.append(dy[i])
                # if wt_i didn't make it to the sensitivity graph,
                # you'd append [None].

        return tuple([None] + new_grads)
