from operations import *
import cupy
import math

################################ matrix multiplication ################################
def matmul_cr(dy_old, dy_new, params):
    temp = cp.reshape(dy_old, (-1, dy_old.shape[-1])) @ dy_new.T
    return cp.reshape(temp, dy_old.shape[:-1] + temp.shape[-1:])

################################ element-wise multiplication ################################
def mul_cr(dy_old, dy_new, params):
    return dy_old * dy_new

################################ reshape ################################
def reshape_cr(dy_old, dy_new, params):
    leading_dim = dy_old.shape[:(1 - dy_new.ndim)]
    return cp.reshape(dy_old, leading_dim + tuple(params[0]))

################################ unsqueeze ################################
def unsqueeze_cr(dy_old, dy_new, params):
    extra_dim = dy_old.ndim - dy_new.ndim + 1
    return cp.expand_dims(X, axis=tuple(params[0] + extra_dim))

################################ transpose ################################
def transpose_cr(dy_old, dy_new, params):
    leading_dim = dy_old.shape[:(1 - dy_new.ndim)]
    # optional parameter given
    if len(params) > 0:
        return cp.transpose(X, axes=(X.shape[0], ) + tuple(params[0] + 1))
    return cp.transpose(X)

################################ argmax ################################
def squeeze_cr(dy_old, dy_new, params):
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

def identity(X, params):
    return X
