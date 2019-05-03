import torch
import torch.nn as nn

import shift

def convert_to_shift(model, shift_depth):
    for name, module in reversed(list(model.named_modules())):
        if type(module) == nn.Linear:
            model._modules[name] = shift.LinearShift(module.in_features, module.out_features, module.bias is not None) 

    return model