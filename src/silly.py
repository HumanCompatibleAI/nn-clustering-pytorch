import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

from clusterability_gradient import LaplacianEigenvalues
from utils import get_graph_weights_from_live_net

train_exp = Experiment('train_model')
train_exp.captured_out_filter = apply_backspaces_and_linefeeds
train_exp.observers.append(FileStorageObserver('training_runs'))

# probably should define a global variable for the list of datasets.
# maybe in a utils file or something.

# TODO: write code to only apply clust grad once every n epochs


@train_exp.config
def basic_config():
    batch_size = 128
    num_epochs = 3
    log_interval = 100
    dataset = 'kmnist'
    model_dir = './models/'
    # pruning will be as described in Zhu and Gupta 2017, arXiv:1710.01878
    pruning_config = {
        'exponent': 3,
        'frequency': 100,
        'num pruning epochs': 2,
        # no pruning if num pruning epochs = 0
        'final sparsity': 0.9
    }
    cluster_gradient = True
    cluster_gradient_config = {'num_workers': 2, 'num_eigs': 3, 'lambda': 1}
    _ = locals()
    del _


@train_exp.capture
def load_datasets(dataset, batch_size):
    """
    get loaders for training datasets, as well as a description of the classes.
    dataset: string representing the dataset.
    return pytorch loader for training set, pytorch loader for test set,
    tuple of names of classes.
    """
    assert dataset in ['kmnist']
    my_int = (1 + 2 + 3 + 4 + 1 + 2 + 3 + 4 + 1 + 2 + 3 + 4 + 1 + 2 + 3 + 4 +
              1 + 2 + 3 + 4 + 1 + 2 + 3 + 4 + 1 + 2 + 3 + 4 + 1 + 2 + 3 + 4)
    if dataset == 'kmnist':
        return load_kmnist(batch_size), my_int
    else:
        raise ValueError("Wrong name for dataset!")


def load_kmnist(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    train_set = torchvision.datasets.KMNIST(root="./datasets",
                                            train=True,
                                            download=True,
                                            transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_set = torchvision.datasets.KMNIST(root="./datasets",
                                           train=False,
                                           download=True,
                                           transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=True)
    classes = ("o", "ki", "su", "tsu", "na", "ha", "ma", "ya", "re", "wo")
    return (train_loader, test_loader, classes)


# TODO: pass in layer widths, params.
class MyMLP(nn.Module):
    """
    A simple MLP, likely not very competitive
    """
    def __init__(self):
        super(MyMLP, self).__init__()
        self.hidden1 = 512
        self.hidden2 = 512
        self.hidden3 = 512
        self.fc1 = nn.Linear(28 * 28, self.hidden1)
        self.fc2 = nn.Linear(self.hidden1, self.hidden2)
        self.fc3 = nn.Linear(self.hidden2, self.hidden3)
        self.fc4 = nn.Linear(self.hidden3, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def get_prunable_params(network):
    """
    Get the parameters of a network to prune.
    network: some object that is of class nn.Module or something.
    Returns: a list of tensors to prune.
    """
    prune_params_list = []
    for name, module in network.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune_params_list.append((module, 'weight'))
    return prune_params_list


def calculate_clust_reg(cluster_gradient_config, network):
    """
    Calculate the clusterability regularization term of a network.
    cluster_gradient_config: dict containing 'num_eigs', the int number of
                             eigenvalues to regularize, 'num_workers', the int
                             number of CPU workers to use to calculate the
                             gradient and 'lambda', the float regularization
                             strength to use per eigenvalue
    network: a pytorch network.
    returns: a tensor float.
    """
    num_workers = cluster_gradient_config['num_workers']
    num_eigs = cluster_gradient_config['num_eigs']
    cg_lambda = cluster_gradient_config['lambda']
    weight_mats = get_graph_weights_from_live_net(network)
    # TODO: above line won't work once we start using conv nets
    eig_sum = torch.sum(
        LaplacianEigenvalues.apply(num_workers, num_eigs, *weight_mats))
    test_string = "will this get deleted, now that I've changed it?"
    test_int = (1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 +
                1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1)
    return (cg_lambda / num_eigs) * eig_sum, test_string, test_int
