import cupy as cp

# convolution layer
def conv(X, params):
    # 1D convolution
    if len(params[0].shape) == 3:
        return conv1d(X, params)
    # 2D convolution
    else:
        return conv2d(X, params)
    
    
# 1D convolution
def conv1d(X, params):
    # extract attributes
    w, b, group, pad, autopad, strides, dilations, kernel_size = \
        params[0], \
        params[1], \
        params[2], \
        params[3], \
        params[4], \
        params[5], \
        params[6], \
        params[0].shape[-1:]
    
    # add padding
    X_padded = cp.pad(
        X,
        pad_width=((0,0), (0,0), (pad[0], pad[1])),
        mode='constant'
    )
    dilated_kernel_size = ((kernel_size[0] - 1) * dilations[0] + 1, )

    # extract parameters
    batch_size = X_padded.shape[0]
    in_channels = X_padded.shape[1]
    sequence_shape = X_padded.shape[2:]

    # generate sub matrix
    new_shape = (batch_size, in_channels) \
        + tuple(map(lambda x, y: (x - y) // strides[0] + 1, sequence_shape, dilated_kernel_size)) \
        + tuple(kernel_size)
    new_strides = X_padded.strides[:2] \
        + (X_padded.strides[2] * strides[0], X_padded.strides[2] * dilations[0])
    submat = cp.repeat(cp.swapaxes(\
              cp.lib.stride_tricks.as_strided(X_padded, new_shape, new_strides), 1, 2)[:, :, None, :, :], \
              w.shape[0], axis=2)

    return cp.swapaxes(cp.sum((w * submat), axis=(3, 4)) + b, 1, 2)

# 2D convolution
def conv2d(X, params):
    # extract attributes
    w, b, dilations, group, kernel_size, pad, strides = \
        params[0], \
        params[1], \
        params[2], \
        params[3], \
        params[4], \
        params[5], \
        params[6]

    # add padding
    X_padded = cp.pad(
        X,
        pad_width=((0,0), (0,0), (pad[0], pad[2]), (pad[1], pad[3])),
        mode='constant'
    )
    dilated_kernel_size = tuple(map(lambda x, y: (x - 1) * y + 1, kernel_size, dilations))

    # extract parameters
    batch_size = X_padded.shape[0]
    in_channels = X_padded.shape[1]
    sequence_shape = X_padded.shape[2:]

    # generate sub matrix
    new_shape = (batch_size, in_channels) \
        + tuple(map(lambda x, y, z: (x - y) // z + 1, sequence_shape, dilated_kernel_size, strides)) \
        + tuple(kernel_size)
    new_strides = X_padded.strides[:2] \
        + (X_padded.strides[2] * strides[0], X_padded.strides[3] * strides[1], \
          X_padded.strides[2] * dilations[0], X_padded.strides[3] * dilations[1])

    submat = cp.repeat(cp.transpose(\
              cp.lib.stride_tricks.as_strided(X_padded, new_shape, new_strides), (0, 2, 3, 1, 4, 5))[:, :, :, None, :, :], \
              w.shape[0], axis=3)

    return cp.transpose(cp.sum((w * submat), axis=(4, 5, 6)) + b, (0, 3, 1, 2)) 


# ReLU
def relu(X, params):
    return cp.maximum(X, 0)

# linear layer
def linear(X, params):
    w, b = params[0], params[1]
    return (w @ X.T).T + b

# reshape
def reshape(x, params):
    return cp.reshape(x, x.shape[0:-3] + tuple([2*14*14]))