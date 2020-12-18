import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

train_exp = Experiment('train_model')
train_exp.captured_out_filter = apply_backspaces_and_linefeeds
train_exp.observers.append(FileStorageObserver('training_runs'))


@train_exp.config
def basic_config():
    batch_size = 128
    num_epochs = 5
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
    if dataset == 'kmnist':
        return load_kmnist(batch_size)
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


def train_and_save(network, train_loader, test_loader, num_epochs,
                   pruning_config, optimizer, criterion, log_interval, device,
                   model_path_prefix, _run):
    """
    Train a neural network, printing out log information, and saving along the
    way.
    network: an instantiated object that inherits from nn.Module or something.
    train_loader: pytorch loader for the training dataset
    test_loader: pytorch loader for the testing dataset
    num_epochs: int for the total number of epochs to train for (including any
                pruning)
    pruning_config: dict containing 'exponent', a numeric type, 'frequency', int
                    representing the number of training steps to have between
                    prunes, 'num pruning epochs', int representing the number of
                    epochs to prune for, and 'final sparsity', float
                    representing how sparse the final net should be
    optimizer: pytorch optimizer. Might be SGD or Adam.
    criterion: loss function.
    log_interval: int. how many training steps between logging infodumps.
    device: pytorch device - do things go on CPU or GPU?
    model_path_prefix: string containing the relative path to save the model.
                       should not include final '.pth' suffix.
    returns: tuple of test acc, test loss, and list of tuples of
             (epoch number, iteration number, train loss)
    """
    network.to(device)
    loss_list = []

    prune_exp = pruning_config['exponent']
    prune_freq = pruning_config['frequency']
    num_pruning_epochs = pruning_config['num pruning epochs']
    final_sparsity = pruning_config['final sparsity']
    is_pruning = num_pruning_epochs != 0
    current_density = 1.
    start_pruning_epoch = num_epochs - num_pruning_epochs
    num_prunes_total = (num_pruning_epochs * len(train_loader)) // prune_freq
    num_prunes_so_far = 0
    train_step_counter = 0

    prune_params_list = []
    for name, module in network.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune_params_list.append((module, 'weight'))

    for epoch in range(num_epochs):
        print("Start of epoch", epoch)
        network.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % log_interval == log_interval - 1:
                avg_loss = running_loss / log_interval
                print(f"batch {i}, avg loss {avg_loss}")
                _run.log_scalar("training.loss", avg_loss)
                loss_list.append((epoch, i, avg_loss))
                running_loss = 0.0
            if epoch >= start_pruning_epoch:
                if train_step_counter % prune_freq == 0:
                    new_sparsity = (
                        final_sparsity *
                        (1 -
                         (1 -
                          (num_prunes_so_far / num_prunes_total))**prune_exp))
                    sparsity_factor = 1 - (
                        (1. - new_sparsity) / current_density)
                    prune.global_unstructured(
                        prune_params_list,
                        pruning_method=prune.L1Unstructured,
                        amount=sparsity_factor)
                    current_density *= 1 - sparsity_factor
                    num_prunes_so_far += 1
                train_step_counter += 1

        test_acc, test_loss = eval_net(network, test_loader, device, criterion,
                                       _run)
        print("Test accuracy is", test_acc)
        print("Test loss is", test_loss)
        if is_pruning and epoch == start_pruning_epoch - 1:
            model_path = model_path_prefix + '_unpruned.pth'
            torch.save(network.state_dict(), model_path)
            train_exp.add_artifact(model_path)

    if is_pruning:
        for module, name in prune_params_list:
            prune.remove(module, name)
    save_path = model_path_prefix + '.pth'
    torch.save(network.state_dict(), save_path)
    train_exp.add_artifact(save_path)
    return test_acc, test_loss, loss_list


def eval_net(network, test_loader, device, criterion, _run):
    """
    gets test loss and accuracy
    network: network to get loss of
    test_loader: pytorch loader of test set
    device: device to put data on
    criterion: loss function
    returns: tuple of floats. first is test accuracy, second is test loss.
    """
    correct = 0.0
    total = 0
    loss_sum = 0.0
    num_batches = 0

    network.eval()
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = network(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            loss_sum += loss.item()
            num_batches += 1

    test_accuracy = correct / total
    test_avg_loss = loss_sum / num_batches
    _run.log_scalar("test.accuracy", test_accuracy)
    _run.log_scalar("test.loss", test_avg_loss)
    return test_accuracy, test_avg_loss


@train_exp.automain
def run_training(dataset, num_epochs, batch_size, log_interval, model_dir,
                 pruning_config, _run):
    """
    Trains and saves network.
    dataset: string specifying which dataset we're using
    num_epochs: int
    batch_size: int
    log_interval: int. number of iterations to go between logging infodumps
    model_dir: string. relative path to directory where model should be saved
    """
    device = (torch.device("cuda")
              if torch.cuda.is_available() else torch.device("cpu"))

    criterion = nn.CrossEntropyLoss()
    my_net = MyMLP()
    optimizer = optim.Adam(my_net.parameters())
    train_loader, test_loader, classes = load_datasets()
    save_path_prefix = model_dir + dataset
    # TODO: come up with better way of generating save_path_prefix
    # or add info as required
    # probably partly do in config
    test_acc, test_loss, loss_list = train_and_save(
        my_net, train_loader, test_loader, num_epochs, pruning_config,
        optimizer, criterion, log_interval, device, save_path_prefix, _run)
    return {
        'test acc': test_acc,
        'test loss': test_loss,
        'train loss list': loss_list
    }
