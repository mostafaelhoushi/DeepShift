import torch
import torch.nn as nn
import numpy as np
import math
import copy

import deepshift.modules
import deepshift.modules_q
import deepshift.utils as utils

def convert_to_shift(model, shift_depth, shift_type, convert_all_linear=True, convert_weights=False, freeze_sign = False, use_kernel=False, use_cuda=True, rounding='deterministic', weight_bits=5):
    conversion_count = 0
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name], num_converted = convert_to_shift(model=module, shift_depth=shift_depth-conversion_count, shift_type=shift_type, convert_all_linear=convert_all_linear, convert_weights=convert_weights, freeze_sign = freeze_sign, use_kernel=use_kernel, use_cuda = use_cuda, rounding = rounding, weight_bits = weight_bits)
            conversion_count += num_converted
        if type(module) == nn.Linear and (convert_all_linear == True or conversion_count < shift_depth):
            linear = module
        
            if shift_type == 'Q':
                shift_linear = deepshift.modules_q.LinearShiftQ(module.in_features, module.out_features, module.bias is not None, use_kernel=use_kernel, use_cuda=use_cuda, rounding=rounding, weight_bits=weight_bits) 
                shift_linear.weight = linear.weight
                if linear.bias is not None:
                    shift_linear.bias.data = utils.round_to_fixed(linear.bias, fraction_bits=16, integer_bits=16)

                if use_cuda==True and use_kernel == True:
                    shift_linear.conc_weight = utils.compress_bits(*utils.get_shift_and_sign(linear.weight))
            elif shift_type == 'PS':
                shift_linear = deepshift.modules.LinearShift(module.in_features, module.out_features, module.bias is not None, freeze_sign = freeze_sign, use_kernel=use_kernel, use_cuda=use_cuda, rounding=rounding, weight_bits=weight_bits)

                if convert_weights == True:
                    shift_linear.shift.data, shift_linear.sign.data = utils.get_shift_and_sign(linear.weight)
                    shift_linear.bias = linear.bias
                
                    if use_cuda==True and use_kernel == True:
                        shift_linear.conc_weight = utils.compress_bits(shift_linear.shift.data, shift_linear.sign.data)
            else:
                raise ValueError('Unsupported shift_type argument: ', shift_type)

            model._modules[name] = shift_linear
            if convert_all_linear == False:
                conversion_count += 1

        if type(module) == nn.Conv2d and conversion_count < shift_depth:
            conv2d = module

            if shift_type == 'Q':
                shift_conv2d = deepshift.modules_q.Conv2dShiftQ(module.in_channels, module.out_channels, module.kernel_size, module.stride,
                                                module.padding, module.dilation, module.groups,
                                                module.bias is not None, module.padding_mode, 
                                                use_kernel=use_kernel, use_cuda=use_cuda,
                                                rounding=rounding, weight_bits=weight_bits) 
                shift_conv2d.weight = conv2d.weight
                if conv2d.bias is not None:
                    shift_conv2d.bias.data = utils.round_to_fixed(conv2d.bias, fraction_bits=16, integer_bits=16)

                if use_cuda==True and use_kernel == True:
                    shift_conv2d.conc_weight = utils.compress_bits(*utils.get_shift_and_sign(conv2d.weight))

            elif shift_type == 'PS':
                shift_conv2d = deepshift.modules.Conv2dShift(module.in_channels, module.out_channels, module.kernel_size, module.stride,
                                                module.padding, module.dilation, module.groups,
                                                module.bias is not None, module.padding_mode,
                                                freeze_sign=freeze_sign, use_kernel=use_kernel, use_cuda=use_cuda,
                                                rounding=rounding, weight_bits=weight_bits)

                if convert_weights == True:
                    shift_conv2d.shift.data, shift_conv2d.sign.data = utils.get_shift_and_sign(conv2d.weight)
                    shift_conv2d.bias = conv2d.bias

                if use_cuda==True and use_kernel == True:
                    shift_conv2d.conc_weight = utils.compress_bits(shift_conv2d.shift.data, shift_conv2d.sign.data)
               
            model._modules[name] = shift_conv2d
            conversion_count += 1

    return model, conversion_count

def round_shift_weights(model, clone=False):
    if(clone):
        model = copy.deepcopy(model)

    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = round_shift_weights(model=module)

        if type(module) == deepshift.modules.LinearShift or type(module) == deepshift.modules.Conv2dShift:
            module.shift.data = module.shift.round()
            module.sign.data = module.sign.round().sign()

            if (module.bias is not None):
                module.bias.data = utils.round_to_fixed(module.bias, fraction_bits=16, integer_bits=16)
        elif type(module) == deepshift.modules_q.LinearShiftQ or type(module) == deepshift.modules_q.Conv2dShiftQ:
            module.weight.data = utils.round_power_of_2(module.weight)

            if (module.bias is not None):
                module.bias.data = utils.round_to_fixed(module.bias, fraction_bits=16, integer_bits=16)

    return model

def count_layer_type(model, layer_type):
    count = 0
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            # recurse
            count += count_layer_type(model=module, layer_type=layer_type)
        if type(module) == layer_type:
            count += 1

    return count    