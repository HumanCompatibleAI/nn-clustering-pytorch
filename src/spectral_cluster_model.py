# Based on code written by Shlomi Hod, Stephen Casper, and Daniel Filan

import numpy as np
import torch
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
from sklearn.cluster import SpectralClustering

from datasets import load_datasets
from graph_utils import (
    add_activation_gradients,
    delete_isolated_ccs,
    normalize_weights_array,
    np_layer_array_to_graph_weights_array,
    weights_to_graph,
)
from networks import CNN_DICT, MLP_DICT
from train_model import csordas_get_input
from utils import (
    load_activations_numpy,
    masked_weights_from_state_dicts,
    model_weights_from_state_dict_numpy,
    np_layer_array_to_bn_param_dicts,
)

clust_exp = Experiment('cluster_model')
clust_exp.captured_out_filter = apply_backspaces_and_linefeeds
clust_exp.observers.append(FileStorageObserver('clustering_runs'))


@clust_exp.config
def basic_config():
    num_clusters = 4
    weights_path = "./models/mlp_kmnist.pth"
    use_activations = False
    acts_load_path = None
    dataset = None
    net_str = None
    mask_path = None
    net_type = 'mlp'
    normalize_weights = True
    epsilon = 1e-9
    eigen_solver = 'arpack'
    _ = locals()
    del _


@clust_exp.named_config
def cnn_config():
    net_type = 'cnn'
    _ = locals()
    del _


@clust_exp.named_config
def act_grad_config():
    use_activations = True
    dataset = 'kmnist'
    net_str = 'mnist'
    _ = locals()
    del _


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
    n-cut is as defined in "Normalized Cuts and Image Segmentation", Shi and
    Malik, 2000.
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
        n_cut_terms[label] = cut / (vol + epsilon)
    return sum(n_cut_terms.values())


def adj_mat_to_clustering_and_quality(adj_mat, num_clusters, eigen_solver,
                                      epsilon):
    """
    Clusters a graph and computes the n-cut.
    adj_mat: sparse square adjacency matrix
    num_clusters: int representing the number of clusters to cluster into
    eigen_solver: string or None representing which eigenvalue solver to use
                  for spectral clustering
    epsilon: small positive float to stop us dividing by zero
    verbose: boolean determining whether we learn the n-cut term for each
             cluster label
    returns a tuple: first element is a float that is the n-cut value, second
                     element is a list of labels of each node
    """
    clustering_labels = cluster_adj_mat(num_clusters, adj_mat, eigen_solver)
    n_cut_val = compute_n_cut(adj_mat, clustering_labels, epsilon)
    return (n_cut_val, clustering_labels)


def layer_array_to_clustering_and_quality(layer_array, net_type, acts_dict,
                                          num_clusters, eigen_solver,
                                          normalize_weights, epsilon):
    """
    Take an array of weight tensors, delete any isolated connected components,
    cluster the resulting graph, and get the n-cut and the clustering.
    layer_array: array of dicts representing layers, containing layer names
                 and numpy param tensors
    acts_dict: dictionary of numpy tensors holding activation data for each
               neuron, or None if that isn't being used.
    num_clusters: integer number of desired clusters
    eigen_solver: string specifying which eigenvalue solver to use for spectral
                  clustering
    normalize_weights: bool specifying whether the weights should be
                       'normalized' before clustering.
    epsilon: small positive number to stop us dividing by zero
    returns: tuple containing n-cut and array of cluster labels
    """
    weights_array = np_layer_array_to_graph_weights_array(
        layer_array, net_type)
    bn_params = np_layer_array_to_bn_param_dicts(layer_array, net_type)
    if normalize_weights:
        weights_array = normalize_weights_array(weights_array)
    if acts_dict is not None:
        weights_array = add_activation_gradients(weights_array, acts_dict,
                                                 net_type, bn_params)
    adj_mat = weights_to_graph(weights_array)
    _, adj_mat_, _, _, isolation_indicator = delete_isolated_ccs(
        weights_array, adj_mat)
    result = adj_mat_to_clustering_and_quality(adj_mat_, num_clusters,
                                               eigen_solver, epsilon)
    return result, isolation_indicator


def get_activations(net_type, net_str, state_dict, dataset):
    """
    Load a dictionary of neuron activations on a specified training dataset.
    Only works for CachingNets as defined in src/networks.py.
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
    _, test_set_dict, _ = load_datasets(dataset, 128, {})
    new_batch = next(iter(test_set_dict['all']))
    net.eval()
    with torch.no_grad():
        if dataset != 'add_mul':
            _ = net(new_batch[0])
        else:
            _ = net(csordas_get_input(new_batch))
        acts_dict = net.activations
    for key, val in acts_dict.items():
        acts_dict[key] = val.detach().cpu().numpy()
    return acts_dict


@clust_exp.automain
def run_experiment(weights_path, mask_path, use_activations, acts_load_path,
                   net_type, num_clusters, eigen_solver, normalize_weights,
                   dataset, net_str, epsilon):
    """
    load saved weights, delete any isolated connected components, cluster them,
    get their n-cut and the clustering
    weights_path: path to where weights are saved. String suffices.
    mask_path: path to where masks are saved, if any, or None.
    use_activations: bool indicating whether to use activation-aware
        clustering.
    acts_load_path: string indicating where to load activations from, or None
        if activations will not be loaded
    net_type: string indicating whether the model is an MLP or a CNN
    num_clusters: int, number of groups to cluster the net into
    eigen_solver: string specifying which eigenvalue solver to use for spectral
        clustering
    normalize_weights: bool specifying whether the weights should be
        'normalized' before clustering.
    dataset: string specifying the dataset, or None if activations will not
        be used.
    net_str: string specifying the network architecture, or None if activations
        will not be used.
    epsilon: small positive number to stop us dividing by zero
    returns: tuple containing n-cut and array of cluster labels
    """
    device = (torch.device("cuda")
              if torch.cuda.is_available() else torch.device("cpu"))
    state_dict = torch.load(weights_path, map_location=device)
    if mask_path is not None:
        mask_dict = torch.load(mask_path, map_location=device)
        state_dict = masked_weights_from_state_dicts(state_dict, mask_dict)
    layer_array = model_weights_from_state_dict_numpy(state_dict)
    if use_activations:
        assert net_str is not None
        assert dataset is not None
    if use_activations:
        acts_dict = (get_activations(net_type, net_str, state_dict, dataset)
                     if acts_load_path is None else load_activations_numpy(
                         acts_load_path, device))
    else:
        acts_dict = None
    (n_cut_val, clustering_labels), isolation_indicator = \
        layer_array_to_clustering_and_quality(layer_array, net_type, acts_dict,
                                              num_clusters, eigen_solver,
                                              normalize_weights, epsilon)
    return n_cut_val, clustering_labels.tolist(), isolation_indicator, [
        i.tolist() for i in np.unique(clustering_labels, return_counts=True)
    ]
