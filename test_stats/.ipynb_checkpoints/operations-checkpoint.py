import cupy as cp
import math

################################ ReLU ################################
def relu(X, params):
    return cp.maximum(X, 0)

################################ Linear (Gemm) ################################
def linear(X, params):
    w, b = params[0], params[1]
    temp = cp.matmul(cp.reshape(X, (-1, X.shape[-1])), w.T) + b
    return cp.reshape(temp, X.shape[:-1] + temp.shape[-1:])
    #return cp.matmul(X, w.T) + b

################################ reshape ################################
def reshape(X, params):
    return cp.reshape(X, (X.shape[0], ) + tuple(params[0]))

################################ batched normalization ################################
def batch_norm(X, params):
    scale, b, epsilon, momentum = \
        params[0], \
        params[1], \
        params[2]

    input_mean = cp.mean(X, axis=0, keepdims=True)
    input_var =  cp.var(X, axis=0, keepdims=True)

    return (X - input_mean) / cp.sqrt(input_var + epsilon) * scale + b

################################ argmax ################################
def argmax(X, params):
    return cp.argmax(X, axis=-1)

################################ unsqueeze ################################
def unsqueeze(X, params):
    return cp.expand_dims(X, axis=tuple(params[0] + 1))

################################ transpose ################################
def transpose(X, params):
    # optional parameter given
    if len(params) > 0:
        return cp.transpose(X, axes=(X.shape[0], ) + tuple(params[0] + 1))
    return cp.transpose(X)

################################ argmax ################################
def squeeze(X, params):
    # optional parameter given
    if len(params) > 0:
        return cp.squeeze(X, axes=tuple(params[0] + 1))
    else:
        axes_to_squeeze = [
            i
            for i, dim in enumerate(X.shape)
            if i > 1 and dim == 1
        ]
        return cp.squeeze(X, axes=axes_to_squeeze)

################################ add ################################
def add(X, params):
    return cp.add(X, params[0])

################################ multiplication ################################
def mul(X, params):
    return cp.multiply(X, params[0])

################################ matrix multiplication ################################
def matmul(X, params):
    # parameter as first operand
    if len(params) > 1:
        temp = cp.matmul(params[0], cp.reshape(X, (-1, X.shape[-1])))
    else:
        temp = cp.matmul(cp.reshape(X, (-1, X.shape[-1])), params[0])
    return cp.reshape(temp, X.shape[:-1] + temp.shape[-1:])

################################ maxpool layer ################################
def maxpool(X, params):
    # 1D maxpool
    return conv1d(X, params)

################################ 1D maxpool layer ################################
def maxpool1d(X, params):
    # unpack parameters assuming a specific order and type.
    # params: (storage_order, pads=(p_start, p_end), ceil_mode, auto_pad, strides, dilations, kernel_size_val)
    _storage_order, _pads, _ceil_mode, _autopad, _strides, _dilations, _kernel_size = params

    # input shape: (Batch_size, Channels, Input_length)
    N, C, L_in = X.shape
    k = _kernel_size
    s = _strides
    d = _dilations
    p_start, p_end = _pads # Assuming pads is a tuple/list like [pad_left, pad_right]

    # effective kernel size considering dilation
    k_eff = (k - 1) * d + 1

    # padding
    if p_start > 0 or p_end > 0:
        X_padded = cp.pad(X, ((0, 0), (0, 0), (p_start, p_end)), mode="constant", constant_values=-cp.inf)
    else:
        X_padded = X

    L_padded = X_padded.shape[-1]

    L_out = (L_padded - k_eff) // s + 1
    stride_N, stride_C, stride_L = X_padded.strides

    # new shape for strided view
    new_shape = (N, C, L_out, k)

    # define new strides for the strided view:
    #   - For N and C dimensions, keep original strides.
    #   - For L_out dimension, stride by 's' (actual stride between windows).
    #   - For kernel dimension 'k', stride by 'd' (dilation rate within a window).
    new_strides = (stride_N, stride_C, stride_L * s, stride_L * d)

    # Create the strided view (no data copy, just a different view of memory)
    submat = cp.lib.stride_tricks.as_strided(X_padded, shape=new_shape, strides=new_strides)

    # Perform max pooling over the last dimension (the kernel dimension)
    return cp.max(submat, axis=-1)

################################ convolution layer ################################
def conv(X, params):
    # 1D convolution
    if len(params[0].shape) == 3:
        return conv1d(X, params)
    # 2D convolution
    else:
        return conv2d(X, params)

################################ 1D convolution layer #################################
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

################################ 2D convolution layer #################################
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
