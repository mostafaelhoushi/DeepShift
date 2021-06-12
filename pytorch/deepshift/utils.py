import torch
import numpy as np
import math

import deepshift.kernels

def round_to_fixed(input, integer_bits=16, fraction_bits=16): 
    assert integer_bits >= 1, integer_bits
    # TODO: Deal with unsigned tensors where there is no sign bit
    #       which is the case with activations to convolution that 
    #       are usually the output of a Relu layer
    if integer_bits == 1: 
        return torch.sign(input) - 1 
    delta = math.pow(2.0, -(fraction_bits))
    bound = math.pow(2.0, integer_bits-1) 
    min_val = - bound 
    max_val = bound - 1 
    rounded = torch.floor(input / delta) * delta

    clipped_value = torch.clamp(rounded, min_val, max_val)
    return clipped_value 

def get_shift_and_sign(x, rounding='deterministic'):
    sign = torch.sign(x)
    
    x_abs = torch.abs(x)
    shift = round(torch.log(x_abs) / np.log(2), rounding)
    
    return shift, sign    

def round_power_of_2(x, rounding='deterministic'):
    shift, sign = get_shift_and_sign(x, rounding)    
    x_rounded = (2.0 ** shift) * sign
    return x_rounded

def round(x, rounding='deterministic'):
    assert(rounding in ['deterministic', 'stochastic'])
    if rounding == 'stochastic':
        x_floor = x.floor()
        return x_floor + torch.bernoulli(x - x_floor)
    else:
        return x.round()

def clampabs(input, min, max):
    assert(min >= 0 and max >=0)

    input[input > max] = max
    input[input < -max] = -max

    input[(input > torch.zeros_like(input)) & (input < min)] = min
    input[(input < torch.zeros_like(input)) & (input > -min)] = -min
    return input 

class ConcWeight():
    def __init__(self, data=None, base=0, bits=8):
        self.data = data 
        self.base = base
        self.bits = bits

##concatenate shift and sign together
def compress_bits(shift, sign):
    conc_weight = ConcWeight() 

    if len(shift.shape) == 2:
        shift = shift.unsqueeze(-1).unsqueeze(-1)

    # if sign is ternary, then use a big shift value that is equivalent to multiplying by zero
    zero_sign_indices = (sign == 0).nonzero()
    shift[zero_sign_indices] = -32
    sign[zero_sign_indices] = +1

    conc_weight.bits = math.ceil(torch.log( - torch.min(shift) + 1)/ np.log(2))
    # treat shift to the right as the default
    shift = shift * -1
    minimum = int(torch.min(shift))
    if minimum < 0:
        conc_weight.base = minimum
        shift = shift - minimum
    else:
        conc_weight.base = 0

    num = int(32 / (conc_weight.bits + 1))
    row_length = int((shift.shape[1] * shift.shape[2] * shift.shape[3] + num -1) / num )
    size = row_length * shift.shape[0]

    conc_weight.data = deepshift.kernels.compress_sign_and_shift(shift.int().cuda(), sign.int().cuda(), size, conc_weight.base, conc_weight.bits, row_length, num)

    return conc_weight
