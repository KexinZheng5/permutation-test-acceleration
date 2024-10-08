import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from time import time
import numpy as np

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
DATA = "./_Data/"
PERMUTED = "./_Data/parameters/permuted/"

trainset = datasets.MNIST(DATA + 'train', download=True, train=True, transform=transform)
testset = datasets.MNIST(DATA + 'test', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, 15)
        self.fc1 = nn.Linear(2 * 14 * 14, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 2*14*14)
        x = self.fc1(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# permutation test
for i in range(100):
    print("permutation", i)
    model = CNN()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # train
    for epoch in range(6):  # loop over the dataset multiple times
        for j, data in enumerate(trainloader, 0):
            inputs, labels = data
            ind = torch.randperm(labels.shape[0])
            labels = labels[ind]
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    # save parameters
    torch.save(model.conv1.weight, PERMUTED+'conv_w' + str(i) + '.pt')
    torch.save(model.fc1.weight, PERMUTED+'fc_w' + str(i) + '.pt')
    torch.save(model.conv1.bias, PERMUTED+'conv_b' + str(i) + '.pt')
    torch.save(model.fc1.bias, PERMUTED+'fc_b' + str(i) + '.pt')