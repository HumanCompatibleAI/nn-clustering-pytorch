import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

batch_size = 128
num_classes = 10
epochs = 1
log_interval = 100

device = (torch.device("cuda")
          if torch.cuda.is_available() else torch.device("cpu"))

# train and test sets
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
        self.fc4 = nn.Linear(self.hidden3, num_classes)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


criterion = nn.CrossEntropyLoss()
my_net = MyMLP()
optimizer = optim.SGD(my_net.parameters(), lr=0.001, momentum=0.9)
my_net.to(device)
my_net.train()

for epoch in range(epochs):
    for i, data in enumerate(train_loader):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = my_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

path = './test_net.pth'
torch.save(my_net.state_dict(), path)
