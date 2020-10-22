# Based on code written by Shlomi Hod, Stephen Casper, and Daniel Filan

import numpy as np
import scipy.sparse as sparse
import pickle

# config variables
num_clusters = 4
weights_path = "does/not/exist.pckl"

# main code

def mlp_weights_to_layer_widths(weights_array):
    """
    take in an array of weight matrices, and return how wide each layer of the
    network is
    """
    layer_widths = [x.shape[0] for x in weights_array]
    layer_widths.append(weights_array[-1].shape[1])
    return layer_widths

my_array = [ np.zeros((4,3)), np.zeros((3,10)), np.zeros((10,20)), np.zeros((20,10))]
