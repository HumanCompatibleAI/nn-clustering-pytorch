import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from add_mul import AddMul

SIMPLE_FUNCTIONS = {
    "high_freq_waves":
    [lambda x: torch.sin(5 * x), lambda x: torch.cos(5 * x)],
    "many_fns":
    [torch.sin, lambda x: torch.log(x + 5 + 1e-3), torch.cos, torch.exp],
    "many_fns_norm": [
        lambda x: torch.sin(x) / 0.72605824,
        lambda x: torch.log(x + 5 + 1e-3) / 0.996789,
        lambda x: torch.cos(x) / 0.6598921, lambda x: torch.exp(x) / 29.747263
    ],
}

DATASETS = [
    'mnist', 'kmnist', 'cifar10', 'tiny_dataset', 'add_mul', 'simple_dataset'
]


def load_datasets(dataset, batch_size, simple_math_config):
    """
    get loaders for training datasets, as well as a description of the classes.

    dataset (str): string representing the dataset.
    batch_size (int): how many things should be in a batch
    simple_math_config (dict): dictionary of config items for simple math
        dataset

    return pytorch loader for training set, pytorch loader for test set,
    tuple of names of classes.
    """
    assert dataset in DATASETS
    if dataset == 'mnist':
        return load_mnist(batch_size)
    elif dataset == 'kmnist':
        return load_kmnist(batch_size)
    elif dataset == 'cifar10':
        return load_cifar10(batch_size)
    elif dataset == 'tiny_dataset':
        return load_tiny_dataset(batch_size)
    elif dataset == 'add_mul':
        return load_add_mul(batch_size)
    elif dataset == "simple_dataset":
        fns = SIMPLE_FUNCTIONS[simple_math_config['fns_name']]
        return load_simple(fns, batch_size, simple_math_config)
    else:
        raise ValueError("Wrong name for dataset!")


def load_mnist(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.1307], [0.3081])])
    train_set = torchvision.datasets.MNIST(root="./datasets",
                                           train=True,
                                           download=True,
                                           transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_set = torchvision.datasets.MNIST(root="./datasets",
                                          train=False,
                                          download=True,
                                          transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=True)
    classes = ('0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine')
    return (train_loader, {'all': test_loader}, classes)


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
    return (train_loader, {'all': test_loader}, classes)


def load_cifar10(batch_size):
    normalize = transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))
    train_set = torchvision.datasets.CIFAR10(
        root="./datasets",
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(), normalize
        ]))
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_set = torchvision.datasets.CIFAR10(
        root="./datasets",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=True)
    classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse",
               "ship", "truck")
    return train_loader, {'all': test_loader}, classes


def load_tiny_dataset(batch_size):
    train_set = TinyDataset()
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=False)
    test_set = TinyDataset()
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=False)
    classes = ("0", "1")
    return train_loader, {'all': test_loader}, classes


def load_add_mul(batch_size):
    train_set = AddMul("train", 100_000, 2)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    valid_set_iid = AddMul("valid", 10_000, 2)
    iid_loader = torch.utils.data.DataLoader(valid_set_iid,
                                             batch_size=batch_size,
                                             shuffle=True)
    valid_set_add = AddMul("valid", 10_000, 2, restrict=["add"])
    add_loader = torch.utils.data.DataLoader(valid_set_add,
                                             batch_size=batch_size,
                                             shuffle=True)
    valid_set_mul = AddMul("valid", 10_000, 2, restrict=["mul"])
    mul_loader = torch.utils.data.DataLoader(valid_set_mul,
                                             batch_size=batch_size,
                                             shuffle=True)
    my_dict = {"all": iid_loader, "add": add_loader, "mul": mul_loader}
    return train_loader, my_dict, ()


def load_simple(fns, batch_size, simple_math_config):
    input_type = simple_math_config['input_type']
    assert input_type in ["single", "streamed"]
    lim = simple_math_config['lim']
    num_batches_train = simple_math_config['num_batches_train']
    num_batches_test = simple_math_config['num_batches_test']
    train_loader = SimpleDataLoader(fns, input_type, lim, batch_size,
                                    num_batches_train)
    test_loader = SimpleDataLoader(fns, input_type, lim, batch_size,
                                   num_batches_test)
    return train_loader, {"all": test_loader}, tuple()


# Custom Data Loader to randomly generate data for simple networks
class SimpleDataLoader:
    def __init__(self,
                 fns,
                 input_type="single",
                 lim=5,
                 batch_size=250,
                 num_batches=500):
        """
        fns ([Tensor(Float)->Tensor(Float)]): List of functions to be
            approximated. Each acts element wise on a tensor
        input_type (str): One of "single", "streamed". Single means a single
            input x and output f1(x), f2(x),..., streamed means a different
            input for each function
        lim (Float): Inputs will be output uniformly between -lim and lim

        Each batch is generated iid at random, and this terminates after
        num_batches batches are requested
        """
        self.fns = fns
        self.input_type = input_type
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.out = len(fns)
        self.lim = lim
        self.count = 0

    def __len__(self):
        return self.num_batches * self.batch_size

    def __iter__(self):
        self.reset()
        return self

    def reset(self):
        self.count = 0

    def __next__(self):
        if self.count >= self.num_batches:
            raise StopIteration
        if self.input_type == "single":
            x = (torch.rand(self.batch_size) - 0.5) * 2 * self.lim
            y = torch.stack([fn(x) for fn in self.fns], axis=1)
        elif self.input_type == "streamed":
            x = (torch.rand(self.batch_size, self.out) - 0.5) * 2 * self.lim
            y = torch.stack([self.fns[i](x[:, i]) for i in range(self.out)],
                            axis=1)
        self.count += 1
        return (x, y)


class TinyDataset(Dataset):
    """
    Tiny dataset meant for debugging
    """
    def __init__(self):
        # randomly generated data
        self.xs = np.array([[0.06320405, 0.51371515, 0.04077784],
                            [0.58809062, 0.58997539, 0.31045666],
                            [0.22153995, 0.81825784, 0.31460745],
                            [0.37792006, 0.05979807, 0.35770925],
                            [0.83125488, 0.50243196, 0.57578912],
                            [0.52504822, 0.57349545, 0.00399584],
                            [0.78231748, 0.21112105, 0.92726576],
                            [0.77467497, 0.76202789, 0.63063908],
                            [0.84998826, 0.00393768, 0.64844679],
                            [0.18798294, 0.54178095, 0.60813651]])
        self.ys = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        x_np_array = self.xs[idx]
        x_tens = torch.from_numpy(x_np_array).float()
        y = self.ys[idx]
        y_tens = torch.tensor(y)
        return x_tens, y_tens
