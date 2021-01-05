import itertools

import numpy as np
import scipy.sparse
import torch
from pathos.multiprocessing import ProcessPool
from scipy.sparse.linalg import eigsh
from torch.autograd import Function

from utils import (
    delete_isolated_ccs,
    invert_deleted_neurons_np,
    weights_to_graph,
)

# TODO: make this work beautifully with CNNs


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
    return scipy.sparse.identity(num_rows, format='csr') - result, degree_vec


# NB: we're going to need delete_isolated_ccs to return the layer masks it
# applies because otherwise you can't fatten up the gradient.

# we're also going to be making a pytorch function here that is hopefully
# computable in numpy


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
            degrees_prev_layer = degree_list[pre_sums[layer -
                                                      1]:pre_sums[layer]]**(
                                                          -0.5)
            dy_dL_terms = dy_dL[e_m, pre_sums[layer - 1]:pre_sums[layer]]
            x = np.multiply(abs_weights_to, degrees_prev_layer)
            m_contribution += 0.5 * np.dot(x, dy_dL_terms)
        # Contribution from neurons that m feeds to:
        if layer + 2 < len(pre_sums):
            abs_weights_from = np.abs(mat_list[layer][:, m])
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


def retrieve(x):
    # placeholder function
    pass


class MyEigenvalues(Function):
    """
    A torch autograd Function that takes the eigenvalues of the normalized
    Laplacian of a neural network
    """
    @staticmethod
    def forward(ctx, num_workers, num_eigs, *args):
        # can only use save_for_backward once!!!!! also only saves tensors!!!!
        ctx.save_for_backward(num_workers, num_eigs)
        w_tens_array = [tens.detach().cpu().numpy() for tens in args]
        ctx.save_for_backward(w_tens_array)
        adj_mat_csr = weights_to_graph(w_tens_array)
        my_tup = delete_isolated_ccs(w_tens_array, adj_mat_csr)
        thin_w_array, thin_adj_mat, del_rows, del_cols = my_tup
        ctx.save_for_backward(del_rows, del_cols)
        lap_mat_csr, degree_vec = adj_to_laplacian_and_degs(thin_adj_mat)
        ctx.save_for_backward(degree_vec)
        evals, evecs = eigsh(lap_mat_csr, num_eigs + 1, sigma=-1.0, which='LM')
        evecs = np.transpose(evecs)
        # ^ makes evecs (num eigenvals) * (size of lap mat)
        outers = []
        for i in range(num_eigs):
            outers.append(np.outer(evecs[i + 1], evecs[i + 1]))
        ctx.save_for_backward(outers)
        return torch.from_numpy(evals)

    @staticmethod
    def backward(ctx, dy):
        # problem: how do I return a variable number of outputs?
        # also somehow I have to retrieve the stuff in context.
        my_tup = retrieve(ctx)
        outers, degree_vec, w_tens_array = my_tup
        num_workers, del_rows, del_cols = my_tup
        dy_dL = np.tensordot(dy, outers, [[0], [0]])
        penult_grad = get_dy_dW_np(degree_vec, w_tens_array, dy_dL,
                                   num_workers)
        assert len(del_rows) == len(del_cols)
        penult_len = "penult_grad different length than expected"
        assert len(penult_grad) == len(del_rows), penult_len
        final_grad = []
        for (i, grad) in enumerate(penult_grad):
            fat_grad = invert_deleted_neurons_np(grad, del_rows[i],
                                                 del_cols[i])
            final_grad.append(fat_grad)
        pass
