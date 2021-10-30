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
# TODO: make docstrings exist


@compare_masks_clusters.config
def basic_config():
    weights_path = (
        "./csordas_weights/export/addmul_feedforward_big/b4t2f6os" +
        "/model.pth")
    all_mask_path = (
        "./csordas_weights/export/addmul_feedforward_big/b4t2f6os" +
        "/mask_all.pth")
    mask_path = ("./csordas_weights/export/addmul_feedforward_big/b4t2f6os" +
                 "/mask_add.pth")
    other_mask_paths = [
        "./csordas_weights/export/addmul_feedforward_big/b4t2f6os/mask_mul.pth"
    ]
    run_json_path = "./shuffle_clust_runs/18/run.json"
    _ = locals()
    del _


def get_weights_in_clusters(label_array, weights_layer_array, labels,
                            all_mask_array):
    weights = [my_dict['fc_weights'] for my_dict in weights_layer_array]
    layer_widths = weights_to_layer_widths(weights)
    # take in a fattened-out label array
    # split it up using layer_widths
    # use it to identify masks for each label other than "-1"
    layer_labels = []
    prev_width = 0
    for width in layer_widths:
        layer_labels.append(label_array[prev_width:prev_width + width])
        prev_width += width
    clust_masks = []
    for lab in labels:
        clust_masks.append(
            get_weights_in_cluster(layer_labels, weights, lab, all_mask_array))
    return clust_masks


def get_weights_in_cluster(layer_labels, weights, lab, all_mask_array):
    assert len(layer_labels) == len(weights) + 1
    assert len(weights) == len(all_mask_array)
    assert any([lab in label_list for label_list in layer_labels])
    clust_mask = []
    for i, weight_mat in enumerate(weights):
        bool_mat = np.zeros_like(weight_mat, bool)
        in_labels = np.array(layer_labels[i])
        out_labels = np.array(layer_labels[i + 1])
        bool_mat[np.ix_(out_labels == lab, in_labels == lab)] = True
        bool_mat = np.logical_and(bool_mat, all_mask_array[i]['fc_weights'])
        clust_mask.append(bool_mat)
    return lab, clust_mask


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
        return get_unmasked_neurons(mask_array)
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


def get_intersection_props_neurons(neuron_indicator, label_array, cluster):
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
    return cluster, num_in_cluster, iou, iomask, ioclust


def get_intersection_props_masks(cluster_mask_tup, mask_array):
    cluster_id, cluster_mask_array = cluster_mask_tup
    assert len(cluster_mask_array) == len(mask_array)
    num_in_mask = 0
    num_in_clust = 0
    intersection = 0
    union = 0
    for i in range(len(cluster_mask_array)):
        cluster_mask = cluster_mask_array[i]
        other_mask = mask_array[i]['fc_weights']
        num_in_mask += len(other_mask[other_mask])
        num_in_clust += len(cluster_mask[cluster_mask])
        intersection_mask = np.logical_and(cluster_mask, other_mask)
        intersection += len(intersection_mask[intersection_mask])
        union_mask = np.logical_or(cluster_mask, other_mask)
        union += len(union_mask[union_mask])
    iou = intersection / union
    iomask = intersection / num_in_mask
    ioclust = 0 if num_in_clust == 0 else intersection / num_in_clust
    if iou == 0 and ioclust != 0:
        print("iou:", iou)
        print("ioclust:", ioclust)
        print("intersection:", intersection)
        print("union:", union)
        ValueError("This makes no sense")
    return cluster_id, num_in_clust, iou, iomask, ioclust


def get_mask_proportions(mask_array, all_mask_array):
    assert len(mask_array) == len(all_mask_array)
    num_in_task_and_all_mask = 0
    num_in_all_mask = 0
    for i in range(len(mask_array)):
        task_mask = mask_array[i]['fc_weights']
        all_mask = all_mask_array[i]['fc_weights']
        task_in_all_mask = np.logical_and(task_mask, all_mask)
        num_in_task_and_all_mask += len(task_in_all_mask[task_in_all_mask])
        num_in_all_mask += len(all_mask[all_mask])
    return num_in_task_and_all_mask / num_in_all_mask


def get_labels_from_run_json(run_json_path):
    with open(run_json_path) as f:
        run_dict = json.load(f)
    label_array = run_dict['result']['labels']
    isolation_indicator = run_dict['result']['isolation_indicator']
    for i, val in enumerate(isolation_indicator):
        if val == 1:
            label_array.insert(i, -1)
    return label_array


@compare_masks_clusters.automain
def run_experiment(weights_path, all_mask_path, mask_path, other_mask_paths,
                   run_json_path):
    device = (torch.device("cuda")
              if torch.cuda.is_available() else torch.device("cpu"))
    mask_array = load_model_weights_pytorch(mask_path, device)
    other_mask_arrays = [
        load_model_weights_pytorch(other_path, device)
        for other_path in other_mask_paths
    ]
    all_mask_array = load_model_weights_pytorch(all_mask_path, device)
    print(get_mask_proportions(mask_array, all_mask_array))
    neuron_stats = get_unique_unmasked_neurons(mask_array, other_mask_arrays)

    label_array = get_labels_from_run_json(run_json_path)
    label_set = set(label_array)
    label_set.discard(-1)
    labels = list(label_set)
    # I'm iterating thru labels too many times...
    neuron_ious_iomasks_ioclusts = [
        get_intersection_props_neurons(neuron_stats, label_array, lab)
        for lab in labels
    ]

    weights_layer_array = load_model_weights_pytorch(weights_path, device)
    clust_masks = get_weights_in_clusters(label_array, weights_layer_array,
                                          labels, all_mask_array)
    weight_ious_iomasks_ioclusts = [
        get_intersection_props_masks(clust_mask, mask_array)
        for clust_mask in clust_masks
    ]
    return neuron_ious_iomasks_ioclusts, weight_ious_iomasks_ioclusts
    # return weight_ious_iomasks_ioclusts
