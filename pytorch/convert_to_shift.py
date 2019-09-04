import torch
import torch.nn as nn
import numpy as np

import shift
from shift import round_to_fixed, get_shift_and_sign

def convert_to_shift(model, shift_depth, convert_all_linear=True, convert_weights=False, use_kernel=False,use_cuda=True):
    conversion_count = 0
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name], num_converted = convert_to_shift(model=module, shift_depth=shift_depth-conversion_count, convert_all_linear=convert_all_linear, convert_weights=convert_weights, use_kernel=use_kernel,use_cuda = use_cuda)
            conversion_count += num_converted
        if type(module) == nn.Linear and (convert_all_linear == True or conversion_count < shift_depth):
            linear = module
        
            shift_linear = shift.LinearShift(module.in_features, module.out_features, module.bias is not None, use_kernel=use_kernel,use_cuda = use_cuda) 

            if convert_weights == True:
                shift_linear.shift.data, shift_linear.sign.data = get_shift_and_sign(linear.weight)
                shift_linear.bias = linear.bias

            model._modules[name] = shift_linear
            if convert_all_linear == False:
                conversion_count += 1

        if type(module) == nn.Conv2d and conversion_count < shift_depth:
            conv2d = module
            shift_conv2d = shift.Conv2dShift(module.in_channels, module.out_channels, module.kernel_size, module.stride,
                                             module.padding, module.dilation, module.groups,
                                             module.bias is not None, module.padding_mode,  use_kernel=use_kernel,use_cuda=use_cuda) 

            if convert_weights == True:
                shift_conv2d.shift.data, shift_conv2d.sign.data = get_shift_and_sign(conv2d.weight)
                shift_conv2d.bias = conv2d.bias

            model._modules[name] = shift_conv2d
            conversion_count += 1

    return model, conversion_count

def count_layer_type(model, layer_type):
    count = 0
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            # recurse
            count += count_layer_type(model=module, layer_type=layer_type)
        if type(module) == layer_type:
            count += 1

    return count    