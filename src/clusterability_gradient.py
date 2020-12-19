import itertools

import numpy as np
import scipy.sparse
from pathos.multiprocessing import ProcessPool

# import torch
# from scipy.sparse.linalg import eigsh

# from spectral_cluster_model import (
#     delete_isolated_ccs,
#     weights_to_graph,
# )
# from utils import invert_layer_masks_np

# TODO: how are we getting layer widths?
# TODO: parallelize by neuron layer, not weight matrix layer
# TODO: make this work beautifully with CNNs
# TODO: import delete_isolated_ccs


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


def get_dy_dW_aux(layer, mat_list, degree_list, widths, pre_sums, dy_dL):
    """
    gets cluster gradient for one layer.
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
    returns: a numpy array representing the gradient, with the same shape as
             mat_list[layer]. elements are float32s.
    """
    mat = mat_list[layer]
    grad = np.zeros(shape=mat.shape)
    # grad will be the gradient of the eigenvalue sum with respect to mat
    # we will fill in the elements of grad one-by-one in the following loop
    # (and then a global multiplication afterwards)
    # remember: in pytorch, 0 index is outputs, 1 index is inputs
    # m is going to be the input neuron for the weight, and n is going to be
    # the output neuron for the weight.
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
        abs_weights_from = np.abs(mat[:, m])
        degrees_next_layer = degree_list[pre_sums[layer +
                                                  1]:pre_sums[layer +
                                                              2]]**(-0.5)
        dy_dL_terms = dy_dL[e_m, pre_sums[layer + 1]:pre_sums[layer + 2]]
        x = np.multiply(abs_weights_from, degrees_next_layer)
        m_contribution += 0.5 * np.dot(x, dy_dL_terms)
        m_contribution *= degree_list[e_m]**(-1.5)
        m_contribs.append(m_contribution)

    n_contribs = []
    for n in range(widths[layer + 1]):
        e_n = pre_sums[layer + 1] + n  # index of neuron n in dy_dL
        # e is for "embedding"
        n_contribution = 0
        # contribution from neurons that feed to n:
        abs_weights_to = np.abs(mat[n, :])
        degrees_prev_layer = degree_list[pre_sums[layer]:pre_sums[layer +
                                                                  1]]**(-0.5)
        dy_dL_terms = dy_dL[e_n, pre_sums[layer]:pre_sums[layer + 1]]
        x = np.multiply(abs_weights_to, degrees_prev_layer)
        n_contribution += 0.5 * np.dot(x, dy_dL_terms)
        # contribution from neurons that n feeds to:
        if layer + 3 < len(pre_sums):
            abs_weights_from = np.abs(mat_list[layer + 1][:, n])
            degrees_next_layer = degree_list[pre_sums[layer +
                                                      2]:pre_sums[layer +
                                                                  3]]**(-0.5)
            dy_dL_terms = dy_dL[e_n, pre_sums[layer + 2]:pre_sums[layer + 3]]
            x = np.multiply(abs_weights_from, degrees_next_layer)
            n_contribution += 0.5 * np.dot(x, dy_dL_terms)
        n_contribution *= degree_list[e_n]**(-1.5)
        n_contribs.append(n_contribution)

    for m, n in itertools.product(range(widths[layer]),
                                  range(widths[layer + 1])):
        e_m = pre_sums[layer] + m
        e_n = pre_sums[layer] + n
        m_val = m_contribs[m]
        n_val = n_contribs[n]
        gradient_term = m_val + n_val
        gradient_term -= (((degree_list[e_m] * degree_list[e_n])**(-0.5)) *
                          dy_dL[e_m, e_n])
        grad[n, m] = gradient_term
    # now, we need to point-wise multiply with sign(mat) to back-prop thru
    # the absolute value function that takes the weights to the adj mat.
    grad = np.multiply(grad, np.sign(mat))
    return grad.astype(np.float32)


def get_dy_dW_np(degree_list, mat_list, dy_dL, num_workers=1):
    """
    Calculates the clusterability gradient of all the weight tensors.
    Takes in:
    degree_list (array-like), which is an array of degrees of each node.
    adj_mat (sparse array), the adjacency matrix. should have shape
    (len(degree_list), len(degree_list)).
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
    num_mats = len(mat_list)
    num_neurons_off = (
        "Different ways of reckoning the number of neurons give" +
        " different results")
    assert num_neurons == len(degree_list), num_neurons_off
    assert num_neurons == dy_dL.shape[0], num_neurons_off
    assert num_neurons == dy_dL.shape[1], num_neurons_off
    with ProcessPool(nodes=num_workers) as p:
        grad_list = p.map(get_dy_dW_aux, range(num_mats),
                          [mat_list] * num_mats, [degree_list] * num_mats,
                          [widths] * num_mats, [pre_sums] * num_mats,
                          [dy_dL] * num_mats)
    return grad_list


# NB: we're going to need delete_isolated_ccs to return the layer masks it
# applies because otherwise you can't fatten up the gradient.

# we're also going to be making a pytorch function here that is hopefully
# computable in numpy
