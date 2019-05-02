import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import math
import numpy as np

'''
# Inherit from Function
class LinearShiftFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    
    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias
'''

class LinearShift(nn.Module):
    def __init__(self, input_features, output_features, bias=True, check_grad=False):
        super(LinearShift, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        if check_grad:
            tensor_constructor = torch.DoubleTensor # double precision required to check grad
        else:
            tensor_constructor = torch.Tensor # In PyTorch torch.Tensor is alias torch.FloatTensor
        self.shift = nn.Parameter(tensor_constructor(output_features, input_features))
        self.sign = nn.Parameter(tensor_constructor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(tensor_constructor(output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        # Not a very smart way to initialize weights
        self.shift.data.uniform_(-10, -1) # (-0.1, 0.1)
        self.sign.data.uniform_(-1, 0) # (-0.1, 0.1)
        if bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        '''
        return LinearShiftFunction.apply(input, self.shift, self.bias)
        '''

        '''
        output = input.mm((2**self.shift * (-1)**self.sign).t())
        if self.bias is not None:
            output += self.bias.unsqueeze(0).expand_as(output)
        return output
        '''
        weight = (2 ** self.shift.round()) * ( (-1) ** self.sign.round() )
        return F.linear(input, weight, self.bias)
        

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

'''
class LinearShift(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(LinearShift, self).__init__(*kargs, **kwargs)

        self.weight.data.uniform_(-8, 8)

    def forward(self, input):

        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        #self.weight.data=self.weight.org.round()
        print("data:")
        print(self.weight.data)
        print("org:")
        print(self.weight.org)
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.expand_as(out)

        return out
'''


linear_shift = LinearShift(20, 30, check_grad=True) # LinearShiftFunction.apply 

from torch.autograd import gradcheck
# gradcheck takes a tuple of tensors as input, check if your gradient
# evaluated with these tensors are close enough to numerical
# approximations and returns True if they all verify this condition.
data = torch.randn(20,20,dtype=torch.double,requires_grad=True)
weight = torch.randn(30,20,dtype=torch.double,requires_grad=True)
input = (data, weight)
test = gradcheck(linear_shift, data, eps=1e-6, atol=1e-4)
print("gradcheck result for linear_shift: ", test)
