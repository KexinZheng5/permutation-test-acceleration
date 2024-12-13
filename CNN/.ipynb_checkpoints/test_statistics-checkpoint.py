import numpy as np
import torch
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
DATA = "/u/home/k/kzheng/permutation-test-acceleration/CNN/_Data/"
PERMUTED = "/u/home/k/kzheng/permutation-test-acceleration/CNN/_Data/parameters/permuted/"
TEST_STAT = "/u/home/k/kzheng/permutation-test-acceleration/CNN/_Data/perm_test_stats/"
trainset = datasets.MNIST(DATA + 'train', download=True, train=True, transform=transform)

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

def calculateT(trainset_np, weights, bias, functions, derivatives, types):
    T = np.zeros(28*28)

    for data in trainset_np:
        pd = partialDerivative(data, weights, bias, functions, derivatives, types)
        T += np.square(pd)
        
    return T / len(trainset_np)

# calculate partial derivative
def partialDerivative(X, w, b, f, df, types):
    y = f[0](X, w[0], b[0])
    dy = df[0](X, w[0], b[0], y)
    for k in range(1, len(f)):
        y_new = f[k](y, w[k], b[k])
        if types[k] == 0:
            dy = f[k](dy)
        elif types[k] == 1:
            dy = df[k](y, w[k], b[k], y_new) * dy
        elif types[k] == 2:
            dy = df[k](dy, y)
        else:
            dy = df[k](y, w[k], b[k], y_new) @ dy.T
        y = y_new
        
    return dy

# 2D convolution
def conv2D(X, w, b):
    c, _, m, n = w.shape
    new_shape = tuple(np.subtract(X.shape, (m,n)) + 1) + (m,n)
    sub_mat = np.lib.stride_tricks.as_strided(X, new_shape, X.strides + X.strides)
    x_new = np.zeros((c, new_shape[0], new_shape[1]))
    for i in range(c):
        x_new[i] = np.sum((w[i] * sub_mat), axis=(2,3)) + b[i]
    return x_new

# ReLU
def relu(X, w=None, b=None):
    return np.maximum(X, 0)

def linear(X, w, b):
    return (w @ X.T).flatten() + b

# reshape function
def reshape(x, w=None, b=None):
    return np.reshape(x, (-1,2*14*14))

def dconv(X, w, b, y):
    kernel_shape = w.shape[2:]
    y_shape = np.subtract(X.shape, w.shape[2:]) + 1
    sub_mat = np.zeros(w.shape[:1] + tuple(np.add((y_shape-1)*2, kernel_shape)))
    sub_mat[:,y_shape[0]-1 : X.shape[0], y_shape[1]-1 : X.shape[1]] = np.reshape(w, tuple([w.shape[0]]) + w.shape[2:])
    grad = np.lib.stride_tricks.as_strided(sub_mat, X.shape + w.shape[:1] + tuple(y_shape), sub_mat.strides[1:] + sub_mat.strides)
    grad = np.flip(grad, axis=(3,4))
    
    return grad

# derivative of ReLU layer
def drelu(x, w, b, y):
    return 1. * (x > 0)

# derivative of linear layer
def dlinear(x, w, b, y):
    return w

# derivative of max function
# dy: partial derivative of the previous iteration
# y: output of the previous iteration
def dmax(dy, y):
    maximum = y[0]
    ind = 0
    for k in range(1, len(y)):
        if y[k] > maximum:
            maximum = y[k]
            ind = k
    return dy[ind]


# permutation test
for i in range(1,100):
    
    conv_w, fc_w, conv_b, fc_b = loadParam(i)
    
    # parameter initialization
    weights = [conv_w, None, None, fc_w, None]
    bias = [conv_b, None, None, fc_b, None]
    functions = [conv2D, relu, reshape, linear, np.max]
    derivatives = [dconv, drelu, None, dlinear, dmax]
    types = [1,1,0,3,2]
    trainset_np = trainset.data.numpy()
    
    T = calculateT(trainset_np, weights, bias, functions, derivatives, types)
    
    # save parameters
    np.savetxt(TEST_STAT + 'stats_' + str(i) + '.txt', T, delimiter=',')