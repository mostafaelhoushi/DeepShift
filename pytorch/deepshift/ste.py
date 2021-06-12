import torch
from torch.autograd import Function
import deepshift.utils as utils

class RoundPowerOf2(Function):
    @staticmethod 
    def forward(ctx, input, stochastic=False):
        return utils.round_power_of_2(input, stochastic)

    @staticmethod 
    def backward(ctx, grad_output):
        return grad_output, None
        
def round_power_of_2(input, stochastic=False):
    return RoundPowerOf2.apply(input, stochastic)

class RoundFixedPoint(Function):
    @staticmethod 
    def forward(ctx, input, act_integer_bits=16, act_fraction_bits=16):
        return utils.round_to_fixed(input, act_integer_bits, act_fraction_bits)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

def round_fixed_point(input, act_integer_bits=16, act_fraction_bits=16):
    return RoundFixedPoint.apply(input, act_integer_bits, act_fraction_bits)

class RoundFunction(Function):
    @staticmethod 
    def forward(ctx, input, rounding='deterministic'):
        return utils.round(input, rounding)
        
    @staticmethod 
    def backward(ctx, grad_output):
        return grad_output, None

def round(input, rounding='deterministic'):
    return RoundFunction.apply(input, rounding)

class SignFunction(Function):
    @staticmethod 
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod 
    def backward(ctx, grad_output):
        return grad_output

def sign(input):
    return SignFunction.apply(input)

class ClampFunction(Function):
    @staticmethod 
    def forward(ctx, input, min, max):
        return torch.clamp(input, min, max)

    @staticmethod 
    def backward(ctx, grad_output):
        return grad_output, None, None

def clamp(input, min, max):
    return ClampFunction.apply(input, min, max)

class ClampAbsFunction(Function):
    @staticmethod
    def forward(ctx, input, min, max):
        assert(min >= 0 and max >=0)

        input[input > max] = max
        input[input < -max] = -max

        input[(input > torch.zeros_like(input)) & (input < min)] = min
        input[(input < torch.zeros_like(input)) & (input > -min)] = -min
        return input 

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

def clampabs(input, min, max):
    return ClampAbsFunction.apply(input, min, max)

class LogFunction(Function):
    @staticmethod 
    def forward(ctx, input):
        return torch.log(input)

    @staticmethod 
    def backward(ctx, grad_output):
        return grad_output

def log(input):
    return LogFunction.apply(input)

class UnsymmetricGradMulFunction(Function):
    @staticmethod 
    def forward(ctx, input1, input2):
        ctx.save_for_backward(input1, input2)
        return torch.mul(input1, input2)

    @staticmethod 
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors
        return grad_output*input2, grad_output

def unsym_grad_mul(input1, input2):
    return UnsymmetricGradMulFunction.apply(input1, input2)


class AbsFunction(Function):
    @staticmethod 
    def forward(ctx, input):
        return torch.abs(input)

    @staticmethod 
    def backward(ctx, grad_output):
        return grad_output

def abs(input):
    return AbsFunction.apply(input)
