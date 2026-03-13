import cupy as cp
import math

################################ ReLU ################################
def drelu(X, params):
    return 1. * (X > 0)

################################ Linear (Gemm) ################################
def dlinear(X, params):
    return params[0]

################################ batched normalization ################################
def dbatch_norm(X, params):
    scale, b, epsilon, momentum = \
        params[0], \
        params[1], \
        params[2]

    return 1 / cp.sqrt(cp.var(X, axis=0, keepdims=True) + epsilon) * scale + b

################################ argmax ################################
def dargmax(dy_old, dy_new, params):
    idx = dy_new.reshape(dy_new.shape + (1,) * (len(dy_old.shape) - len(dy_new.shape)))
    return cp.squeeze(cp.take_along_axis(dy_old, idx, axis=-1), axis=-1)

################################ add ################################
def dadd(X, params):
    return X

################################ multiplication ################################
def dmul(X, params):
    return params[0]

################################ matrix multiplication ################################
def dmatmul(X, params):
    return params[0].T

################################ maxpool layer ################################
def dmaxpool(X, params):
    # 1D maxpool
    return conv1d(X, params)

################################ 1D maxpool layer ################################
def dmaxpool1d(X, params):
    # Unpack parameters
    _storage_order, _pads, _ceil_mode, _autopad, _strides, _dilations, _kernel_size = params

    # Input shape: (Batch_size, Channels, Input_length)
    N, C, L_in = X.shape
    k = _kernel_size
    s = _strides
    p_start, p_end = _pads
    d = _dilations

    # Compute effective kernel size (must match maxpool1d)
    k_eff = (k - 1) * d + 1

    # Apply padding to the input
    if p_start > 0 or p_end > 0:
        x_pad = np.pad(X, ((0,0),(0,0),(p_start, p_end)),
                       mode="constant", constant_values=-np.inf)
    else:
        x_pad = X

    L_pad = x_pad.shape[-1]

    # Compute output length (must match maxpool1d)
    L_out = (L_pad - k_eff) // s + 1

    # Get strides of the padded input array
    stride_N, stride_C, stride_L = x_pad.strides

    # Define the new shape for the strided view: (N, C, L_out, k)
    new_shape = (N, C, L_out, k)

    # Define new strides for the strided view:
    #   - For N and C dimensions, keep original strides.
    #   - For L_out dimension, stride by 's' (actual stride between windows).
    #   - For kernel dimension 'k', stride by 'd' (dilation rate within a window).
    new_strides = (stride_N, stride_C, stride_L * s, stride_L * d)

    # Create the strided view of all possible windows (no data copy)
    windows_view = np.lib.stride_tricks.as_strided(x_pad, shape=new_shape, strides=new_strides)

    # Find the maximum value in each window along the kernel dimension
    max_vals = windows_view.max(axis=-1, keepdims=True) # Shape (N, C, L_out, 1)

    # Create the mask: 1 where the element equals the max, 0 otherwise
    return (windows_view == max_vals).astype(np.float32)


################################ convolution layer ################################
def dconv(X, params):
    # 1D convolution
    if len(params[0].shape) == 3:
        return dconv1d(X, params)
    # 2D convolution
    else:
        return dconv2d(X, params)

################################ 1D convolution layer #################################
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

################################ 2D convolution layer #################################
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
