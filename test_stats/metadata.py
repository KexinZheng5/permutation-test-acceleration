from operations import *
from gradients import *
from chain_rules import *

################################ operations ################################
op = {
    "Conv" : conv,
    "Relu" : relu,
    "Reshape" : reshape,
    "Gemm" : linear,
    "Argmax" : argmax,
    "BatchNormalization": batch_norm,
}

################################ gradient for each operation ################################
grad = {
    "Conv" : dconv,
    "Relu" : drelu,
    "Reshape" : identity,
    "Gemm" : dlinear,       
    "Argmax" : argmax,
    "BatchNormalization": dbatch_norm
}

################################ defines how chain rule is applied ################################
rule = {
    "Conv" : matmul_cr,
    "Relu" : mul_cr,
    "Reshape" : reshape_cr,
    "Gemm" : matmul_cr,
    "Argmax" : dargmax,
    "BatchNormalization": matmul_cr
}

################################ indicates if the operation gradient can be cached ################################
# Note: cacheable gradients are independent of the input, so there is no addtional dimension indicating the batch size (for memory efficiency)
cacheable = {
    "Conv" : True,
    "Relu" : False,
    "Reshape" : False,
    "Gemm" : True,
    "Argmax" : False,
    "BatchNormalization": False
}
