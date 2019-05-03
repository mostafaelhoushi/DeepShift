import torch
import torch.nn as nn

import shift

def convert_to_shift(model, shift_depth):
    conversion_count = 0
    for name, module in reversed(list(model.named_modules())):
        if type(module) == nn.Linear and conversion_count < shift_depth:
            model._modules[name] = shift.LinearShift(module.in_features, module.out_features, module.bias is not None) 
            conversion_count += 1

    return model