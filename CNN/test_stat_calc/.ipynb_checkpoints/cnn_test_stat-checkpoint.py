from util import *
from operations import *
from derivatives import *

def calculateT(batch_size):
    # get parameters
    conv_w, fc_w, conv_b, fc_b = loadParam()

    conv_w = cp.asarray(conv_w)
    fc_w = cp.asarray(fc_w)
    conv_b = cp.asarray(conv_b)
    fc_b = cp.asarray(fc_b)
    X = cp.asarray(trainset.data.numpy())
    
    # insert axis for in-channel size
    X = X[:, None, :, :]

    # cache for storing reusable partial derivatives
    cache = {}
    
    # initialize test statistics
    T = cp.zeros(math.prod(X[0].shape))

    # for each batch, calculate the partial derivatices and test statistics
    for i in range(0, X.shape[0] - 1, batch_size):
        #start_time = time.time()
        pd = pd_batch_org(cache, X[i:min(i+batch_size, X.shape[0])], conv_w, fc_w, conv_b, fc_b)
        #print(i, time.time() - start_time)
        T += cp.sum(cp.square(pd), axis=0)
    
    return T / len(trainset)
       
# calculate the partial derivatives for a batch of input images  
def pd_batch_org(cache, X, conv_w, fc_w, conv_b, fc_b):
    # convolution layer
    y_0 = conv(X, [conv_w, conv_b, [1,1], 0, [15, 15], [0,0,0,0], [1,1]]) # CONV (input size, conv output size)  * cacheable (independent of input)
    if 0 in cache:
        dy_0 = cache[0]
    else:
        dy_0 = dconv(X, [conv_w, conv_b, [1,1], 0, [15, 15], [0,0,0,0], [1,1]])
        cache[0] = dy_0
    
    # relu layer
    y_1 = relu(y_0, None)                 # RELU (batch size, input output size) * not cacheable (dependent of input)
    dy_1 = drelu(y_0, None)
    dy_1 =  cp.broadcast_to(dy_1[:, None, None, None, :, :, :], (X.shape[0], 1, 28, 28, 2, 14, 14)) * dy_0
    
    # reshape
    y_2 = reshape(y_1, None)              # RESHAPE (shape modification only)
    dy_2 = reshape(dy_1, None)
    
    dy_2 = dy_2.reshape(-1, dy_2.shape[-1])
    
    # linear layer
    y_3 = linear(y_2, [fc_w, fc_b])   # LINEAR (weight, number of output) * cacheable (independent of input)
    dy_3 = dlinear(y_2, [fc_w])
    dy_3 = dy_3 @ dy_2.T
    
    # max layer
    dy_4 = dmax(dy_3.T, y_3)
    
    return dy_4.reshape(X.shape[0], -1)


if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    
from cupyx.profiler import benchmark, profile
from cupy.cuda import nvtx

x = cp.random.rand(4096)

with profile():
    y = cp.sin(x)
    cp.cuda.runtime.deviceSynchronize()