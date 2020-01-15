import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from torch.nn import init
import math
import numpy as np
import time
from shift import round_to_fixed, get_shift_and_sign, round_power_of_2

try:
    import shift_kernel
    import shift_cuda_kernel
except:
    print("Unable to import CPU and/or CUDA bit-wise shift kernels")

class STERoundPowerOf2(Function):
    @staticmethod 
    def forward(ctx, input):
        return round_power_of_2(input)

    @staticmethod 
    def backward(ctx, grad_output):
        return grad_output

class STERoundFixedPoint(Function):
    @staticmethod 
    def forward(ctx, input):
        return round_to_fixed(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def ste_round_fixed_point(input):
    return STERoundFixedPoint.apply(input)

def ste_round_power_of_2(input):
    return STERoundPowerOf2.apply(input)

class STERoundFunction(Function):
    @staticmethod 
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod 
    def backward(ctx, grad_output):
        return grad_output

def ste_round(input):
    return STERoundFunction.apply(input)

class STESignFunction(Function):
    @staticmethod 
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod 
    def backward(ctx, grad_output):
        return grad_output

def ste_sign(input):
    return STESignFunction.apply(input)

class STELogFunction(Function):
    @staticmethod 
    def forward(ctx, input):
        return torch.log(input)

    @staticmethod 
    def backward(ctx, grad_output):
        return grad_output

def ste_log(input):
    return STELogFunction.apply(input)

class STEDivFunction(Function):
    @staticmethod 
    def forward(ctx, input, const):
        return torch.div(input, const)

    @staticmethod 
    def backward(ctx, grad_output):
        return grad_output, None

def ste_div(input, const):
    return STEDivFunction.apply(input, const)


class STEAbsFunction(Function):
    @staticmethod 
    def forward(ctx, input):
        return torch.abs(input)

    @staticmethod 
    def backward(ctx, grad_output):
        return grad_output

def ste_abs(input):
    return STEAbsFunction.apply(input)


# Inherit from Function
class LinearShiftQFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None, use_kernel=False, use_cuda=True):
        fraction_bits = 16
        integer_bit = 16

        shift, sign = get_shift_and_sign(weight)
   
        if use_kernel:
            input_fixed_point = (input * (2 ** fraction_bits)).int()
            if bias is not None:
                bias_fixed_point = (bias * (2 ** fraction_bits)).int()

            if(use_cuda):         
                out = torch.zeros([input.size(0), shift.size(0)], dtype=torch.int32, device=torch.device('cuda:0'))
                if bias is not None:
                    shift_cuda_kernel.linear_shift(input_fixed_point, shift.int(), sign.int(), bias_fixed_point, out)
                else:
                    temp = torch.zeros([shift.size(0)], dtype=torch.int32, device=torch.device('cuda:0'))
                    shift_cuda_kernel.linear_shift(input_fixed_point, shift.int(), sign.int(), temp, out)
                out = out.float()
                out = out / (2**fraction_bits)
            else:
                nn = shift_kernel.linear_kernel(input_fixed_point.detach().numpy(), shift.detach().numpy(), sign.detach().numpy(), bias_fixed_point.detach().numpy())
                out = torch.FloatTensor(nn)
                out = out / (2**fraction_bits)
        else:
            input.data = round_to_fixed(input.data, fraction_bits, integer_bit)
            if bias is not None:
                bias.data = round_to_fixed(bias.data, fraction_bits, integer_bit)

            weight_s = (2.0 ** shift) * sign
            out = input.mm(weight_s.t())
            if bias is not None:
                out += bias.unsqueeze(0).expand_as(out)

        ctx.save_for_backward(input, weight_s, bias)

        return out

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight_s, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight_s)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input) # * v * math.log(2)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, None, None

class LinearShiftQ(nn.Module):
    def __init__(self, in_features, out_features, bias=True, check_grad=False, use_kernel=False, use_cuda=True):
 
        super(LinearShiftQ, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_kernel = use_kernel
        self.check_grad = check_grad
        self.use_cuda = use_cuda
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

        self.weight = nn.Parameter(tensor_constructor(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(tensor_constructor(out_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return LinearShiftQFunction.apply(input, self.weight, self.bias, self.use_kernel, self.use_cuda)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

# check gradient of linear_shift
linear_shift = LinearShiftQ(20, 30, check_grad=True) 
#linear_shift = LinearShiftFunction.apply 

from torch.autograd import gradcheck
# gradcheck takes a tuple of tensors as input, check if your gradient
# evaluated with these tensors are close enough to numerical
# approximations and returns True if they all verify this condition.
data = torch.randn(20,20,dtype=torch.double,requires_grad=True)
weight = torch.randn(30,20,dtype=torch.double,requires_grad=True)
input = (data, weight)
# test = gradcheck(linear_shift, data, eps=1e-6, atol=1e-4)
# print("gradcheck result for linear_shift: ", test)

# Inherit from Function
class Conv2dShiftQFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, use_kernel=False, use_cuda=False):
        fraction_bits = 16
        integer_bits = 16

        shift, sign = get_shift_and_sign(weight)

        if use_kernel:
            input_fixed_point = (input * (2 ** fraction_bits)).int()
            if bias is not None:
                bias_fixed_point = (bias * (2 ** fraction_bits)).int()

            if(use_cuda):
                if len(padding) == 2:
                    padding = (padding[0], padding[0], padding[1], padding[1])
                else:
                    padding = padding
                input_fixed_point = F.pad(input = input_fixed_point, pad = padding, mode = 'constant', value = 0)
                if len(stride) == 1:
                    strides_h = stride[0]
                    strides_w = stride[0]
                else: 
                    strides_h = stride[0]
                    strides_w = stride[1]
                out_height = int((input_fixed_point.size(2) - shift.size(2)) / strides_h +1)
                out_width = int((input_fixed_point.size(3) - shift.size(3)) / strides_w +1)
                out = torch.zeros([input_fixed_point.size(0), shift.size(0), out_height, out_width], dtype=torch.int32, device=torch.device('cuda:0'))

                if bias is not None:
                    shift_cuda_kernel.conv2d_shift(input_fixed_point, shift.int(), sign.int(), bias_fixed_point, out, stride, padding)
                else:
                    temp = torch.zeros([shift.size(0)], dtype=torch.int32, device=torch.device('cuda:0'))
                    shift_cuda_kernel.conv2d_shift(input_fixed_point, shift.int(), sign.int(), temp, out, stride, padding)
                out = out.float()
                out = out / (2**fraction_bits)
            else:
                input_fixed_point = F.pad(input = input_fixed_point, pad = padding, mode = 'constant', value = 0)
                out = shift_kernel.convolution_kernel(input_fixed_point.detach().numpy(), 
                    shift.detach().numpy(),
                    sign.detach().numpy(),
                    bias_fixed_point.detach().numpy(), stride, padding)
                out = torch.FloatTensor(out)
                out = out / (2**fraction_bits)
        else:
            input.data = round_to_fixed(input.data, fraction_bits, integer_bits)
            if bias is not None:
                bias.data = round_to_fixed(bias.data, fraction_bits, integer_bits)

            weight_s = (2.0 ** shift) * sign
            out = F.conv2d(input, weight_s, bias, stride, padding, dilation, groups)

        ctx.save_for_backward(input, weight_s, bias)
        ctx.stride = stride
        ctx.padding = padding 
        ctx.dilation = dilation
        ctx.groups = groups

        return out

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight_s, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding 
        dilation = ctx.dilation
        groups = ctx.groups
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, weight_s, grad_output, stride, padding, dilation, groups)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input, weight_s.shape, grad_output, stride, padding, dilation, groups) # * v * math.log(2)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0,2,3)).squeeze(0)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None

class _ConvNdShiftQ(nn.Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias', 'padding_mode']

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode, 
                 check_grad=False):
        super(_ConvNdShiftQ, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        if transposed:
            self.weight = nn.Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

class Conv2dShiftQ(_ConvNdShiftQ):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', 
                 check_grad=False, use_kernel=False,use_cuda =True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.use_kernel = use_kernel
        self.use_cuda = use_cuda
        super(Conv2dShiftQ, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode,
            check_grad)

    #@weak_script_method
    def forward(self, input):       
        weight_q = ste_round_power_of_2(self.weight)
        input_fxied_point = ste_round_fixed_point(input)
        if self.bias is not None:
            bias_fixed_point = ste_round_fixed_point(self.bias)
        else:
            bias_fixed_point = None

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return torch.nn.functional.conv2d(F.pad(input_fxied_point, expanded_padding, mode='circular'),
                                              weight_q, bias_fixed_point, self.stride,
                                              _pair(0), self.dilation, self.groups)
        else:
            return torch.nn.functional.conv2d(input_fxied_point, weight_q, bias_fixed_point, self.stride,
                                              self.padding, self.dilation, self.groups)
