from metadata import *
import onnx
from onnx import numpy_helper, helper, TensorProto
import time


################################ arrays for storing runtime info ################################ 
runtime_info_onnx = []

def get_debug_info_onnx(prev_t):
    cp.cuda.runtime.deviceSynchronize()
    runtime_info_onnx[-1].append(time.perf_counter() - prev_t)
    
################################ calculate test statistics ################################ 
def calc_stats(graph, X, batch_size, debug=False):
    cache = {}  # cache
    params = {} # get parameters
    layer_params = []
    
    extract_params(graph, layer_params, params)
    
    T = cp.zeros(X[0].shape + (1, ))

    # for each batch, calculate the partial derivatices and test statistics
    for i in range(0, X.shape[0] - 1, batch_size):
        start_time = time.time()
        pd = pd_batch(graph.node, cache, layer_params, X[i:min(i+batch_size, X.shape[0])], debug)
        T += cp.sum(cp.square(pd), axis=0)
    
    return T / X.shape[0]


################################ organize parameter by layer ################################ 
def extract_params(graph, layer_params, params):
    # initialize parameter cache
    for init in graph.initializer:
        params[init.name] = cp.asarray(numpy_helper.to_array(init))
    
    # for each operation
    for node in graph.node:
        layer_params.append([])
            
        # store paremeters in list (for each layer)
        for p in node.input:
            if p in params:
                layer_params[-1].append(params[p])
                
        # extract attributes
        for attr in node.attribute:
            layer_params[-1].append(helper.get_attribute_value(attr))
        
        # extract input as constant
        if node.op_type == "Constant":
            params[node.output[0]] = numpy_helper.to_array(node.attribute[0].t)
            continue
        
        # indicate matmul 
        if node.op_type == "Matmul":
            # add dummy parameter if the first operand is the parameter
            if node.input[0] in params:
                layer_params[-1].append(None)
            
            
################################ partial derivative calculation for a batch of input ################################ 
def pd_batch(nodes, cache, params, X, debug=False):
    y_old = X
    dy_old = None
    i = 0
    
    if debug:
        cp.cuda.runtime.deviceSynchronize()
        runtime_info_onnx.append([])
         
    for node in nodes:
        if node.op_type != "Constant": # skip constant layer (already processed during parameter extraction)
            # set timer
            if debug:
                cp.cuda.runtime.deviceSynchronize()
                t = time.perf_counter()
                
            # calculate operation output and operation gradient
            operation = node.op_type
            y_new = op[operation](y_old, params[i])
            
            # obtain gradient from cache if available
            if cacheable[operation]:
                if i not in cache:
                    dy_new = grad[operation](y_old, params[i])
                    cache[i] = dy_new
                else:
                    dy_new = cache[i]
            else:
                dy_new = grad[operation](y_old, params[i])
                
            # chain rule
            if i > 0: # not first layer
                # may need to reshape array by input size before applying chain rule
                if i == 1:
                    target_shape = (dy_new.shape[0], *X.shape[1:], *dy_new.shape[1:])
                    dy_new = cp.broadcast_to(cp.expand_dims(dy_new, axis=tuple(range(1, X.ndim))), target_shape)
                dy_new = rule[operation](dy_old, dy_new, params[i])
            
            # record runtime
            if debug:
                get_debug_info_onnx(t)
            
            # update values
            y_old = y_new
            dy_old = dy_new
        
        i += 1
            
    return dy_old