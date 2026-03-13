from operations import *
import cupy
import math

def matmul_cr(dy_old, dy_new, params):
    temp = cp.reshape(dy_old, (-1, dy_old.shape[-1])) @ dy_new.T
    return cp.reshape(temp, dy_old.shape[:-1] + temp.shape[-1:])

def mul_cr(dy_old, dy_new, params):
    return dy_old * dy_new

def reshape_cr(dy_old, dy_new, params):
    leading_dim = dy_old.shape[:(1 - dy_new.ndim)]
    return cp.reshape(dy_old, leading_dim + tuple(params[0]))

def identity(X, params):
    return X
