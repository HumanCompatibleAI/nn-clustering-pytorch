import json

import numpy as np
import torch
from sacred import Experiment
from sacred.observers import FileStorageObserver

from utils import load_model_weights_pytorch, weights_to_layer_widths

compare_masks_clusters = Experiment('compare_masks_clusters')
compare_masks_clusters.observers.append(
    FileStorageObserver('compare_masks_clusters'))

# NB: so far, this code only works for MLPs
# TODO: make docstrings nicer


@compare_masks_clusters.config
def basic_config():
    weights_path = (
        "./csordas_weights/export/addmul_feedforward_big/b4t2f6os" +
        "/model.pth")
    mask_path = ("./csordas_weights/export/addmul_feedforward_big/b4t2f6os" +
                 "/mask_add.pth")
    other_mask_paths = [
        "./csordas_weights/export/addmul_feedforward_big/b4t2f6os/mask_mul.pth"
    ]
    run_json_path = "./shuffle_clust_runs/109/run.json"
    _ = locals()
    del _


def get_unmasked_neurons(mask_layer_array):
    """
    Take a mask of a neural network. Return a list with length num neurons,
    with 1 if the neuron is attached to an unmasked weight and 0 if it isn't.
    """
    layer_masks = [my_dict['fc_weights'] for my_dict in mask_layer_array]
    layer_widths = weights_to_layer_widths(layer_masks)
    big_array = []
    for layer, width in enumerate(layer_widths):
        neurons = [0 for _ in range(width)]
        for i in range(width):
            if layer != 0:
                weight_to = np.any(layer_masks[layer - 1][i, :])
            else:
                weight_to = False
            if layer != len(layer_widths) - 1:
                weight_from = np.any(layer_masks[layer][:, i])
            else:
                weight_from = False
            connected = weight_to or weight_from
            neurons[i] = 1 if connected else 0
        big_array += neurons
    return big_array


def get_unique_unmasked_neurons(mask_array, background_arrays):
    if background_arrays == []:
        return mask_array
    else:
        mask_indicator = get_unmasked_neurons(mask_array)
        background_indicators = [
            get_unmasked_neurons(background_array)
            for background_array in background_arrays
        ]
        merged_array = []
        for i in range(len(background_indicators[0])):
            is_in_background = any([
                background_indicators[j][i] == 1
                for j in range(len(background_indicators))
            ])
            is_unique = mask_indicator[i] == 1 and not is_in_background
            indicator = 1 if is_unique else 0
            merged_array.append(indicator)
        return merged_array


def get_intersection_props(neuron_indicator, label_array, cluster):
    error_string = ("Label_array has length " + str(len(label_array)) +
                    ", neuron_indicator has length " +
                    str(len(neuron_indicator)))
    assert len(label_array) == len(neuron_indicator), error_string
    num_unmasked_neurons = sum(neuron_indicator)
    cluster_indicator = [1 if x == cluster else 0 for x in label_array]
    num_in_cluster = sum(cluster_indicator)
    product_array = []
    sum_array = []
    for i in range(len(label_array)):
        mask_status = neuron_indicator[i]
        cluster_status = cluster_indicator[i]
        product_array.append(mask_status * cluster_status)
        sum_array.append(min(1, mask_status + cluster_status))
    intersection = sum(product_array)
    union = sum(sum_array)
    iou = intersection / union
    iomask = intersection / num_unmasked_neurons
    ioclust = intersection / num_in_cluster
    return iou, iomask, ioclust


def get_labels_from_run_json(run_json_path):
    with open(run_json_path) as f:
        run_dict = json.load(f)
    label_array = run_dict['result']['labels']
    isolation_indicator = run_dict['result']['isolation_indicator']
    for i, val in enumerate(isolation_indicator):
        if val == 1:
            label_array.insert(i, "-")
    return label_array


@compare_masks_clusters.automain
def run_experiment(weights_path, mask_path, other_mask_paths, run_json_path):
    device = (torch.device("cuda")
              if torch.cuda.is_available() else torch.device("cpu"))
    mask_array = load_model_weights_pytorch(mask_path, device)
    other_mask_arrays = [
        load_model_weights_pytorch(other_path, device)
        for other_path in other_mask_paths
    ]
    neuron_stats = get_unique_unmasked_neurons(mask_array, other_mask_arrays)

    label_array = get_labels_from_run_json(run_json_path)
    label_set = set(label_array)
    label_set.discard("-")
    labels = list(label_set)
    # I'm iterating thru labels too many times...
    ious_iomasks_ioclusts = [
        get_intersection_props(neuron_stats, label_array, lab)
        for lab in labels
    ]
    return ious_iomasks_ioclusts
