import cupy as cp

# gradient of 2D convolution layer
def dconv(X, params):
    # 1D convolution
    if len(params[0].shape) == 3:
        return dconv1d(X, params)
    # 2D convolution
    else:
        return dconv2d(X, params)
    
def dconv1d(X, params):
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

    # extract parameters
    batch_size = X.shape[0]
    in_channels = X.shape[1]
    sequence_shape = (X.shape[2] + pad[0] + pad[1], )

    dilated_kernel_size = ((kernel_size[0] - 1) * dilations[0] + 1, )
    output_shape = (sequence_shape[0] - dilated_kernel_size[0]) // strides[0] + 1

    submat_length = X.shape[2] + (output_shape - 1) * strides[0]
    pad_submat = (output_shape - 1) * strides[0] \
                - (pad[0] - (max(0, math.ceil(pad[0]-dilated_kernel_size[0]) / strides[0]) * strides[0]))

    # create sub matrix
    # dilation
    submat = cp.zeros(w.shape[0:2] + (w.shape[2] + ((dilations[0]-1) * (w.shape[2]-1)), ), dtype=w.dtype)
    submat[:, :, ::dilations[0]] = cp.flip(w, axis=2)

    # padding
    submat = cp.pad(
        submat,
        pad_width=((0,0), (0,0), (submat_length - pad_submat - dilated_kernel_size[0], pad_submat)),
        mode='constant'
    )

    # find gradient
    grad_shape = submat.shape[0:2] + (X.shape[2], output_shape)
    grad_stride = submat.strides[0:2] + (submat.strides[2], submat.strides[2] * strides[0])

    grad = cp.lib.stride_tricks.as_strided(submat, grad_shape, grad_stride)

    return cp.transpose(cp.flip(grad, axis=2), (1,2,0,3))


def dconv2d(X, params):
    # extract attributes
    w, b, dilations, group, kernel_size, pad, strides = \
        params[0], \
        params[1], \
        params[2], \
        params[3], \
        params[4], \
        params[5], \
        params[6]

    # extract parameters
    batch_size = X.shape[0]
    in_channels = X.shape[1]
    sequence_shape = (X.shape[2] + pad[0] + pad[2], X.shape[3] + pad[1] + pad[3])
    if len(kernel_size) == 1:
        kernel_size = (kernel_size[0], kernel_size[0])

    dilated_kernel_size = tuple(map(lambda x, y: (x - 1) * y + 1, kernel_size, dilations))
    output_shape = tuple(map(lambda x, y, z: (x - y) // z + 1, sequence_shape, dilated_kernel_size, strides))

    submat_shape = tuple(map(lambda x, y, z: x + (y - 1) * z, X.shape[2:], output_shape, strides))

    indexes = tuple(map(lambda x, y, z: x - (max(0, math.ceil((x-y)/z))) * z, [pad[0], pad[2]], dilated_kernel_size, strides))
    pad_submat = tuple(map(lambda x, y, z: (x - 1) * y - z, output_shape, strides, indexes))

    # create sub matrix
    # dilation
    submat = cp.zeros(w.shape[0:2] + tuple(map(lambda x, y: (x - 1) * (y - 1) + y, dilations, w.shape[2:])), dtype=w.dtype)
    submat[:, :, ::dilations[0], ::dilations[1]] = cp.flip(w, axis=(2, 3))

    # padding
    submat = cp.pad(
        submat,
        pad_width=((0,0), (0,0)) + tuple(map(lambda x, y, z: (x - y - z, y), submat_shape, pad_submat, dilated_kernel_size)),
        mode='constant'
    )

    # find gradient
    grad_shape = submat.shape[0:2] + X.shape[2:] + output_shape
    grad_stride = submat.strides[0:2] + submat.strides[2:] + tuple(map(lambda x, y: x * y, submat.strides[2:], strides))

    grad = cp.lib.stride_tricks.as_strided(submat, grad_shape, grad_stride)

    return cp.transpose(cp.flip(grad, axis=2), (1,2,3,0,4,5))

# gradient of ReLU layer
def drelu(x, params):
    return 1. * (x > 0)

# gradient of linear layer
def dlinear(x, params):
    return params[0]

# gradient of max function
# dy: partial derivative of the previous iteration
# y: output of the previous iteration
def dmax(dy, y):
    ind = cp.expand_dims(cp.repeat(cp.argmax(y, axis=1), dy.shape[0] // y.shape[0]), axis=1)
    return cp.take_along_axis(dy, ind, axis=-1)