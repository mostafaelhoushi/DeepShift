import torch
from torch.autograd import Function
import deepshift.utils as utils

class RoundPowerOf2(Function):
    @staticmethod 
    def forward(ctx, input):
        return utils.round_power_of_2(input)

    @staticmethod 
    def backward(ctx, grad_output):
        return grad_output
        
def round_power_of_2(input):
    return RoundPowerOf2.apply(input)

class RoundFixedPoint(Function):
    @staticmethod 
    def forward(ctx, input):
        return utils.round_to_fixed(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def round_fixed_point(input):
    return RoundFixedPoint.apply(input)

class RoundFunction(Function):
    @staticmethod 
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod 
    def backward(ctx, grad_output):
        return grad_output

def round(input):
    return RoundFunction.apply(input)

class SignFunction(Function):
    @staticmethod 
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod 
    def backward(ctx, grad_output):
        return grad_output

def sign(input):
    return SignFunction.apply(input)

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