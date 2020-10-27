import torch
import torch.nn as nn
import torch.nn.functional as F
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
    num_classes = 10
    num_epochs = 20
    log_interval = 100
    dataset = 'kmnist'
    model_path = './kmnist_mlp.pth'
    _ = locals()
    del _


# train and test sets
@train_exp.capture
def load_datasets(dataset, batch_size):
    """
    get loaders for training datasets, as well as a description of the classes.
    dataset: string representing the dataset.
    return pytorch loader for training set, pytorch loader for test set,
    and tuple of names of classes.
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
    def __init__(self, num_classes):
        super(MyMLP, self).__init__()
        self.hidden1 = 512
        self.hidden2 = 512
        self.hidden3 = 512
        self.fc1 = nn.Linear(28 * 28, self.hidden1)
        self.fc2 = nn.Linear(self.hidden1, self.hidden2)
        self.fc3 = nn.Linear(self.hidden2, self.hidden3)
        self.fc4 = nn.Linear(self.hidden3, num_classes)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def train_net(network, train_loader, test_loader, num_epochs, optimizer,
              criterion, log_interval, device, _run):
    """
    TODO write docstring
    """
    network.to(device)
    loss_list = []
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
        test_acc, test_loss = eval_net(network, test_loader, device, criterion,
                                       _run)
        print("Test accuracy is", test_acc)
        print("Test loss is", test_loss)
    return test_acc, test_loss, loss_list


def eval_net(network, test_loader, device, criterion, _run):
    """
    TODO write docstring
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
def train_network(dataset, num_classes, num_epochs, batch_size, log_interval,
                  model_path, _run):
    """
    TODO docstring
    """
    device = (torch.device("cuda")
              if torch.cuda.is_available() else torch.device("cpu"))

    criterion = nn.CrossEntropyLoss()
    my_net = MyMLP(num_classes)
    optimizer = optim.Adam(my_net.parameters())
    train_loader, test_loader, classes = load_datasets()
    test_acc, test_loss, loss_list = train_net(my_net, train_loader,
                                               test_loader, num_epochs,
                                               optimizer, criterion,
                                               log_interval, device, _run)
    torch.save(my_net.state_dict(), model_path)
    return {
        'test acc': test_acc,
        'test loss': test_loss,
        'train loss list': loss_list
    }
