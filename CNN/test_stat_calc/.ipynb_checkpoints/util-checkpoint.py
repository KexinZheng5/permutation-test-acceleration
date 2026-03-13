import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import numpy as np
import math

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

DATA = "./_Data/"
ORIGINAL = "./_Data/parameters/original/"
PERMUTED = "./_Data/parameters/permuted/"
TEST_STAT = "./_Data/perm_test_stats/"

trainset = datasets.MNIST(DATA + 'train', download=True, train=True, transform=transform)
testset = datasets.MNIST(DATA + 'test', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

cpu = torch.device('cpu')
# load parameters and convert to np array
def loadParam(num=None):
    # load parameters (original)
    if num == None:
        d1 = ORIGINAL+'conv_w.pt'
        d2 = ORIGINAL+'fc_w.pt'
        d3 = ORIGINAL+'conv_b.pt'
        d4 = ORIGINAL+'fc_b.pt'
    # load parameters (permuted)
    else:
        d1 = PERMUTED+'conv_w' + str(num)+ '.pt'
        d2 = PERMUTED+'fc_w' + str(num)+ '.pt'
        d3 = PERMUTED+'conv_b' + str(num)+ '.pt'
        d4 = PERMUTED+'fc_b' + str(num)+ '.pt'
    
    conv_w = torch.load(d1, map_location=cpu, weights_only=True).detach().cpu().numpy()
    fc_w = torch.load(d2, map_location=cpu, weights_only=True).detach().cpu().numpy()
    conv_b = torch.load(d3, map_location=cpu, weights_only=True).detach().cpu().numpy()
    fc_b = torch.load(d4, map_location=cpu, weights_only=True).detach().cpu().numpy()
     
    return conv_w, fc_w, conv_b, fc_b