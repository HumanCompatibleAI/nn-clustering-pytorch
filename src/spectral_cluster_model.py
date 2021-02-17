# Based on code written by Shlomi Hod, Stephen Casper, and Daniel Filan

import numpy as np
import torch
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
from sklearn.cluster import SpectralClustering

from graph_utils import delete_isolated_ccs, weights_to_graph
from utils import load_model_weights_pytorch

# TODO: extend to CNNs etc.

clust_exp = Experiment('cluster_model')
clust_exp.captured_out_filter = apply_backspaces_and_linefeeds
clust_exp.observers.append(FileStorageObserver('clustering_runs'))


@clust_exp.config
def basic_config():
    num_clusters = 4
    weights_path = "./models/kmnist.pth"
    epsilon = 1e-9
    eigen_solver = 'arpack'
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


def weights_array_to_clustering_and_quality(weights_array, num_clusters,
                                            eigen_solver, epsilon):
    """
    Take an array of weight tensors, delete any isolated connected components,
    cluster the resulting graph, and get the n-cut and the clustering.
    weights_array: array of numpy weight tensors
    num_clusters: integer number of desired clusters
    eigen_solver: string specifying which eigenvalue solver to use for spectral
                  clustering
    epsilon: small positive number to stop us dividing by zero
    returns: tuple containing n-cut and array of cluster labels
    """
    adj_mat = weights_to_graph(weights_array)
    weights_array_, adj_mat_, _, _ = delete_isolated_ccs(
        weights_array, adj_mat)
    return adj_mat_to_clustering_and_quality(adj_mat_, num_clusters,
                                             eigen_solver, epsilon)


@clust_exp.automain
def run_experiment(weights_path, num_clusters, eigen_solver, epsilon):
    """
    load saved weights, delete any isolated connected components, cluster them,
    get their n-cut and the clustering
    weights_path: path to where weights are saved. String suffices.
    num_clusters: int, number of groups to cluster the net into
    eigen_solver: string specifying which eigenvalue solver to use for spectral
                  clustering
    epsilon: small positive number to stop us dividing by zero
    returns: tuple containing n-cut and array of cluster labels
    """
    device = (torch.device("cuda")
              if torch.cuda.is_available() else torch.device("cpu"))
    weights_array_ = load_model_weights_pytorch(weights_path, device)
    return weights_array_to_clustering_and_quality(weights_array_,
                                                   num_clusters, eigen_solver,
                                                   epsilon)
