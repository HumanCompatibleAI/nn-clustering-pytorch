import itertools

import numpy as np
import scipy.sparse
import torch
from pathos.multiprocessing import ProcessPool
from scipy.sparse.linalg import eigsh
from torch.autograd import Function

from graph_utils import (
    delete_isolated_ccs,
    invert_deleted_neurons_np,
    weights_to_graph,
)
from utils import size_and_multiply_np, size_sqrt_divide_np


def adj_to_laplacian_and_degs(adj_mat_csr):
    """
    Takes in:
    adj_mat_csr, a sparse adjacency matrix in CSR format
    Returns:
    a tuple of two elements:
    the first element is the normalised laplacian matrix Lsym in CSR format
    the second element is the degree vector as a numpy array.
    """
    num_rows = adj_mat_csr.shape[0]
    degree_vec = np.squeeze(np.asarray(adj_mat_csr.sum(axis=0)))
    inv_sqrt_degrees = np.reciprocal(np.sqrt(degree_vec))
    inv_sqrt_deg_col = np.expand_dims(inv_sqrt_degrees, axis=1)
    inv_sqrt_deg_row = np.expand_dims(inv_sqrt_degrees, axis=0)
    result = adj_mat_csr.multiply(inv_sqrt_deg_row)
    result = result.multiply(inv_sqrt_deg_col)
    return (((1 + 1e-5) * scipy.sparse.identity(num_rows, format='csr') -
             result), degree_vec)


def get_neuron_contribs_dy_dW(layer, mat_list, degree_list, widths, pre_sums,
                              dy_dL):
    """
    gets cluster gradient contributions from each neuron in one layer
    Takes in:
    layer: an int for which weight tensor we're getting the gradient for
    mat_list: an array of numpy tensors representing weight tensors of the net
    degree_list: an array of the degrees of each node
    widths: an array of the widths of each layer of the network (should be
            ints)
    pre_sums: an array where the ith element is the sum of widths of layers
              before layer i
    dy_dL: a 2d numpy array of floats of the cluster gradient with respect to
           the graph laplacian.
    returns: an array of floats representing neuron contributions to the
             gradient. There should be one entry for every neuron in the layer
    """
    # remember: in pytorch, 0 index is outputs, 1 index is inputs
    # NB: this could be parallelized one day
    m_contribs = []
    for m in range(widths[layer]):
        e_m = pre_sums[layer] + m  # this is the index of neuron m in dy_dL
        # e is for "embedding"
        m_contribution = 0
        # Contribution from neurons that feed to m:
        if layer != 0:
            abs_weights_to = np.abs(mat_list[layer - 1][m, :])
            # sum over any spatial dimensions of conv filters
            if len(abs_weights_to.shape) == 3:
                abs_weights_to = np.sum(abs_weights_to, axis=(1, 2))
            degrees_prev_layer = degree_list[pre_sums[layer -
                                                      1]:pre_sums[layer]]**(
                                                          -0.5)
            dy_dL_terms = dy_dL[e_m, pre_sums[layer - 1]:pre_sums[layer]]
            x = np.multiply(abs_weights_to, degrees_prev_layer)
            m_contribution += 0.5 * np.dot(x, dy_dL_terms)
        # Contribution from neurons that m feeds to:
        if layer + 2 < len(pre_sums):
            abs_weights_from = np.abs(mat_list[layer][:, m])
            # sum over any spatial dimensions of conv filters
            if len(abs_weights_from.shape) == 3:
                abs_weights_from = np.sum(abs_weights_from, axis=(1, 2))
            degrees_next_layer = degree_list[pre_sums[layer +
                                                      1]:pre_sums[layer +
                                                                  2]]**(-0.5)
            dy_dL_terms = dy_dL[e_m, pre_sums[layer + 1]:pre_sums[layer + 2]]
            x = np.multiply(abs_weights_from, degrees_next_layer)
            m_contribution += 0.5 * np.dot(x, dy_dL_terms)
        m_contribution *= degree_list[e_m]**(-1.5)
        m_contribs.append(m_contribution)
    return m_contribs


def get_dy_dW_single(layer, mat, neuron_contribs, degree_list, pre_sums,
                     widths, dy_dL):
    """
    Calculates dy_dW for a single weight tensor.
    Takes in:
    layer: an int for which weight tensor we're getting the gradient for
    mat: the weight tensor that we're getting the gradient for
    neuron_contribs: an array of arrays of floats, representing contribs to the
                     gradient from each neuron. First index is the layer,
                     second is the index of the neuron within the layer.
    degree_list: an array of the degrees of each node (floats)
    pre_sums: an array where the ith element is the sum of widths of layers
              before layer i
    widths: an array of the widths of each layer of the network (should be
            ints)
    returns: a numpy array representing the gradient, with the same shape as
             mat. elements are float32s.
    """
    grad = np.zeros(shape=mat.shape)
    m_contribs = neuron_contribs[layer]
    n_contribs = neuron_contribs[layer + 1]
    for m, n in itertools.product(range(widths[layer]),
                                  range(widths[layer + 1])):
        e_m = pre_sums[layer] + m
        e_n = pre_sums[layer] + n
        m_val = m_contribs[m]
        n_val = n_contribs[n]
        gradient_term = m_val + n_val
        gradient_term -= (((degree_list[e_m] * degree_list[e_n])**(-0.5)) *
                          dy_dL[e_m, e_n])
        # remember: in pytorch, 0 index is outputs, 1 index is inputs
        grad[n, m] = gradient_term
        # for conv layers, this assigns to every filter element.
    grad = np.multiply(grad, np.sign(mat))
    return grad.astype(np.float32)


def get_dy_dW_np(degree_list, mat_list, dy_dL, num_workers=1):
    """
    Calculates the clusterability gradient of all the weight tensors.
    Takes in:
    degree_list (array-like), which is an array of degrees of each node.
    mat_list, a list of numpy arrays of the weight matrices
    dy_dL (rank 2 np ndarray), a num_eigenvalues * len(degree_list)
    * len(degree_list) tensor of the derivatives of the eigenvalue with respect
    to the laplacian entries
    Returns:
    grad_list, an array of numpy arrays representing the derivatives of the
    eigenvalues with respect to the individual weights.
    """
    # we're going to be dividing by degrees later, so none of them can be zero
    assert np.all(degree_list != 0), "Some degrees were zero in get_dy_dW_np!"
    widths = [mat.shape[1] for mat in mat_list]
    widths.append(mat_list[-1].shape[0])
    cumulant = np.cumsum(widths)
    pre_sums = np.insert(cumulant, 0, 0)
    # pre_sums[i] is the number of neurons before layer i
    num_neurons = cumulant[-1]
    num_layers = len(widths)
    num_neurons_off = (
        "Different ways of reckoning the number of neurons give" +
        " different results")
    assert num_neurons == len(degree_list), num_neurons_off
    assert num_neurons == dy_dL.shape[0], num_neurons_off
    assert num_neurons == dy_dL.shape[1], num_neurons_off
    with ProcessPool(nodes=num_workers) as p:
        neuron_contrib_list = p.map(get_neuron_contribs_dy_dW,
                                    range(num_layers), [mat_list] * num_layers,
                                    [degree_list] * num_layers,
                                    [widths] * num_layers,
                                    [pre_sums] * num_layers,
                                    [dy_dL] * num_layers)
    # now get gradient from neuron contribs
    num_mats = len(mat_list)
    with ProcessPool(nodes=num_workers) as p:
        grad_list = p.map(get_dy_dW_single, range(num_mats), mat_list,
                          [neuron_contrib_list] * num_mats,
                          [degree_list] * num_mats, [pre_sums] * num_mats,
                          [widths] * num_mats, [dy_dL] * num_mats)
    return grad_list


def tensor_arrays_to_graph_weights_array(tensor_type_array, tensor_array,
                                         net_type):
    """
    Take tensor_array and tensor_type_array, as defined in train_model.py (but
    numpified), and produce an array of 'weight tensor' equivalents that can be
    turned into a graph.
    tensor_type_array: array of strings saying the role of each tensor
    tensor_array: array of numpy tensors
    net_type: string indicating whether the network is an MLP or a CNN
    returns: array of numpy tensors ready to be turned into a graph
    """
    assert len(tensor_array) == len(tensor_type_array)
    assert net_type in ['mlp', 'cnn']
    weight_name = 'fc_weights' if net_type == 'mlp' else 'conv_weights'
    weights_array = []
    for tens, name in zip(tensor_array, tensor_type_array):
        if name == weight_name:
            weights_array.append(tens)
        if name == 'bn_weights':
            my_tens = weights_array[-1]
            weights_array[-1] = size_and_multiply_np(tens, my_tens)
        if name == 'bn_running_var':
            my_tens = weights_array[-1]
            weights_array[-1] = size_sqrt_divide_np(tens, my_tens)
    return weights_array


class LaplacianEigenvalues(Function):
    """
    A torch autograd Function that takes the eigenvalues of the normalized
    Laplacian of a neural network. These essentially measure how clusterable
    the network is.
    """
    @staticmethod
    def forward(ctx, num_workers, num_eigs, net_type, tensor_type_array,
                *args):
        assert isinstance(num_workers, int)
        assert isinstance(num_eigs, int)
        assert net_type in ['mlp', 'cnn']
        assert isinstance(tensor_type_array, list)
        assert tensor_type_array != []
        for x in tensor_type_array:
            assert isinstance(x, str)

        # this array contains numpy versions of all input tensors
        np_tensor_array = [tens.detach().cpu().numpy() for tens in args]
        # this array contains what the tensors would be if the weights absorbed
        # the batch norm stuff
        w_tens_np_array = tensor_arrays_to_graph_weights_array(
            tensor_type_array, np_tensor_array, net_type)
        num_layers = len(w_tens_np_array)
        adj_mat_csr = weights_to_graph(w_tens_np_array)
        deleted_isolated_ccs = delete_isolated_ccs(w_tens_np_array,
                                                   adj_mat_csr)
        if deleted_isolated_ccs is None:
            # the neural network isn't connected from start to end, it doesn't
            # make sense to regularize any more.
            # it would probably be better to add a bit to every weight and then
            # continue, but that's harder.
            ctx.fully_connected = False
            ctx.num_inputs = len(np_tensor_array)
            return torch.zeros(num_eigs)
        else:
            ctx.fully_connected = True
            (thin_w_array, thin_adj_mat, del_rows, del_cols,
             _) = deleted_isolated_ccs
            assert len(del_rows) == num_layers
            assert len(del_cols) == num_layers
            assert len(thin_w_array) == num_layers
            lap_mat_csr, degree_vec = adj_to_laplacian_and_degs(thin_adj_mat)
            evals, evecs = eigsh(lap_mat_csr, num_eigs + 1, which='SM')
            evecs = np.transpose(evecs)
            # ^ makes evecs (num eigenvals) * (size of lap mat)
            outers = []
            for i in range(num_eigs):
                outers.append(np.outer(evecs[i + 1], evecs[i + 1]))
            ctx.num_workers = num_workers
            ctx.net_type = net_type
            ctx.degree_vec = degree_vec
            ctx.thin_w_np_array = thin_w_array
            ctx.del_rows = del_rows
            ctx.del_cols = del_cols
            ctx.outers = outers
            ctx.tensor_types = tensor_type_array
            ctx.save_for_backward(*args)
            return torch.from_numpy(evals[1:])

    @staticmethod
    def backward(ctx, dy):
        # NB: this is a possibly sketchy way of selecting the device
        device = (torch.device("cuda")
                  if torch.cuda.is_available() else torch.device("cpu"))
        if not ctx.fully_connected:
            # network isn't connected end-to-end, don't bother with gradients
            num_inputs = ctx.num_inputs
            return tuple([None] * (4 + num_inputs))
        else:
            # recover saved tensors
            num_workers = ctx.num_workers
            net_type = ctx.net_type
            assert net_type in ['mlp', 'cnn']
            degree_vec = ctx.degree_vec
            thin_w_np_array = ctx.thin_w_np_array
            del_rows = ctx.del_rows
            del_cols = ctx.del_cols
            outers = ctx.outers
            tensor_types = ctx.tensor_types
            tens_array = ctx.saved_tensors
            # keeping this in case del_rows and del_cols actually have to be
            # lists
            # del_rows = [
            #     entry.detach().cpu().numpy().tolist()
            #     for entry in del_rows_tens
            # ]
            # del_cols = [
            #     entry.detach().cpu().numpy().tolist()
            #     for entry in del_cols_tens
            # ]
            np_tens_array = [
                tens.detach().cpu().numpy() for tens in tens_array
            ]

            # calculate gradient wrt 'thin tensors' (post deletion of isolated
            # ccs)
            # this is where the bulk of the calculation is done!
            dy_dL = np.tensordot(dy, outers, [[0], [0]])
            penult_grad = get_dy_dW_np(degree_vec, thin_w_np_array, dy_dL,
                                       num_workers)
            assert len(del_rows) == len(del_cols)
            penult_len = "penult_grad different length than expected"
            assert len(penult_grad) == len(del_rows), penult_len
            # calculate gradient wrt 'layer tensors'
            fat_grads = []
            for (i, grad) in enumerate(penult_grad):
                fat_grad = invert_deleted_neurons_np(grad, del_rows[i],
                                                     del_cols[i])
                # fat_grads.append(torch.from_numpy(fat_grad).to(device))
                fat_grads.append(fat_grad)
            # calculate gradient wrt all input tensors
            # basically, go thru tensor_types. if you correspond to just a
            # weight matrix, then you stay put. if you correspond to a weight
            # matrix + batch norm stuff, then calculate gradients for all of
            # the stuff.
            # Note for batch norm: the running variances are a function of the
            # weights that you're taking the gradient of, so you'd think you'd
            # have to use the chain rule. But when you work it out, for nets of
            # width n, this is a factor ~1/n correction, and it's hard to work
            # out, so we'll just ignore it.
            # Since here we don't actually know how the weights will update
            # (that depends on the alg (e.g. SGD vs Adam) and the learning
            # rate) we won't manually edit the running variance, and will hope
            # that sorts itself out.
            final_grad = []
            weight_name = ('fc_weights'
                           if net_type == 'mlp' else 'conv_weights')
            grouped_tens_types = []
            assert tensor_types[0] == weight_name
            for i, tens_type in enumerate(tensor_types):
                if tens_type == weight_name:
                    grouped_tens_types.append([weight_name])
                else:
                    grouped_tens_types[-1].append(tens_type)
            # grouped_tens_types should look like
            # [[weight_name], [weight_name], [weight_name, bn_weights,
            # bn_running_var], [weight_name, bn_running_var], [weight_name]]
            assert len(grouped_tens_types) == len(fat_grads)
            # turn gradients into gradients for weights and batch norm stuff
            # separately.
            for (i, grad) in enumerate(fat_grads):
                current_group = grouped_tens_types[i]
                assert current_group[0] == weight_name
                if len(current_group) == 1:
                    final_grad.append(torch.from_numpy(grad).to(device))
                else:
                    np_tens_start_index = sum(
                        [len(group) for group in grouped_tens_types[:i]])
                    tens_dict = {'weights': np_tens_array[np_tens_start_index]}
                    for j in range(1, len(current_group)):
                        tens = np_tens_array[np_tens_start_index + j]
                        if current_group[j] == 'bn_running_var':
                            tens_dict['bn_running_var'] = tens
                        if current_group[j] == 'bn_weights':
                            tens_dict['bn_weights'] = tens
                    assert 'bn_running_var' in tens_dict \
                        or 'bn_weights' in tens_dict
                    weight_grad = grad
                    if 'bn_weights' in tens_dict:
                        weight_grad = size_and_multiply_np(
                            tens_dict['bn_weights'], weight_grad)
                    if 'bn_running_var' in tens_dict:
                        weight_grad = size_sqrt_divide_np(
                            tens_dict['bn_running_var'], weight_grad)
                    final_grad.append(torch.from_numpy(weight_grad).to(device))

                    # add gradients for bn_weights and bn_running_var
                    # bn_running_var gets no gradient.
                    if ('bn_weights' in tens_dict
                            and 'bn_running_var' in tens_dict):
                        bn_w_grad = size_sqrt_divide_np(
                            tens_dict['bn_running_var'], grad)
                        bn_w_grad = np.multiply(bn_w_grad,
                                                tens_dict['weights'])
                        bn_w_grad = np.sum(bn_w_grad,
                                           tuple(range(1, bn_w_grad.ndim)))
                        w_index = current_group.index('bn_weights')
                        v_index = current_group.index('bn_running_var')
                        assert w_index != v_index
                        if w_index < v_index:
                            final_grad.append(
                                torch.from_numpy(bn_w_grad).to(device))
                            final_grad.append(None)
                        else:
                            final_grad.append(None)
                            final_grad.append(
                                torch.from_numpy(bn_w_grad).to(device))
                    elif 'bn_running_var' in tens_dict:
                        final_grad.append(None)
                    else:
                        # now tens_dict just has bn_weights
                        # after writing this code I've realized that this
                        # should theoretically never happen. But whatever.
                        bn_w_grad = np.multiply(grad, tens_dict['weights'])
                        bn_w_grad = np.sum(bn_w_grad,
                                           tuple(range(1, bn_w_grad.ndim)))
                        final_grad.append(
                            torch.from_numpy(bn_w_grad).to(device))

            # finally, gradient for num_workers, num_eigs, net_type, and
            # tensor_type_array should be None.
            # because those aren't differentiable, they're more like
            # parameters.
            return tuple([None, None, None, None] + final_grad)
