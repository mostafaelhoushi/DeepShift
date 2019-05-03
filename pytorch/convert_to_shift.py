import torch
import shift

def convert_to_shift(model, shift_depth):
    model._modules['fc1'] = shift.LinearShift(1*28*28, 512) 

    return model