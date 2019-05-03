import torch
import torch.nn as nn

import shift

def convert_to_shift(model, shift_depth):
    for name, module in model.named_modules():
        if type(module) == nn.Linear:
            model._modules[name] = shift.LinearShift(1*28*28, 512) 

    return model