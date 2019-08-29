import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from torch.nn import init
import shift_kernel
import shift_cuda_kernel
import math
import numpy as np
import time

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):

    # First figure out what the size of the output should be

    N, C, H, W = x_shape

    assert (H + 2 * padding - field_height) % stride == 0

    assert (W + 2 * padding - field_height) % stride == 0

    out_height = (H + 2 * padding - field_height) / stride + 1

    out_width = (W + 2 * padding - field_width) / stride + 1

    out_width = int(out_width)
    out_height = int(out_height)

    i0 = np.repeat(np.arange(field_height), field_width)

    i0 = np.tile(i0, C)

    i1 = stride * np.repeat(np.arange(out_height), out_width)

    j0 = np.tile(np.arange(field_width), field_height * C)

    j1 = stride * np.tile(np.arange(out_width), out_height)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)

    j = j0.reshape(-1, 1) + j1.reshape(1, -1)



    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)



    return (k, i, j)





def im2col_indices(x, field_height, field_width, padding=1, stride=1):

    """ An implementation of im2col based on some fancy indexing """

    # Zero-pad the input

    p = padding

    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')



    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,

                                stride)



    cols = x_padded[:, k, i, j]

    C = x.shape[1]

    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)

    return cols.transpose()

def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,

                   stride=1):

    """ An implementation of col2im based on fancy indexing and np.add.at """

    N, C, H, W = x_shape

    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    print(cols.dtype)
    x_padded = np.zeros((N, C, H_padded, W_padded), int)

    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,

                                stride)

    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)

    cols_reshaped = cols_reshaped.transpose(2, 0, 1)

    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)

    if padding == 0:

        return x_padded

    return x_padded[:, :, padding:-padding, padding:-padding]

def round_to_fixed(input, bits=16):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input) - 1
    delta = math.pow(2.0, -(bits/2))
    bound = math.pow(2.0, bits-1)
    min_val = - bound
    max_val = bound - 1
    rounded = torch.floor(input / delta + 0.5)

    clipped_value = torch.clamp(rounded, min_val, max_val) * delta
    return clipped_value

def get_shift_and_sign(x):
    sign = torch.sign(x)
    # convert sign to (-1)^sign
    # i.e., 1 -> 0, -1 -> 1
    #sign = sign.numpy()
    sign[sign == 1] = 0
    sign[sign == -1] = 1
    
    x_abs = torch.abs(x)
    shift = torch.round(torch.log(x_abs) / np.log(2))

    return shift, sign    

def round_power_of_2(x):
    shift, sign = get_shift_and_sign(x)    
    x_rounded = (2.0 ** shift) * sign
    return x_rounded

class LinearShift(nn.Module):
    def __init__(self, in_features, out_features, bias=True, check_grad=False, use_kernel=False,use_cuda =True):
 
        super(LinearShift, self).__init__()
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
        self.shift = nn.Parameter(tensor_constructor(out_features, in_features))
        self.sign = nn.Parameter(tensor_constructor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(tensor_constructor(out_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        weights = torch.zeros_like(self.shift)
        init.kaiming_uniform_(weights, a=math.sqrt(5))
        self.shift.data, self.sign.data = get_shift_and_sign(weights)
        
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.shift)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
    
        if self.check_grad is False:
            input.data=round_to_fixed(input.data)
            self.bias.data=round_to_fixed(self.bias.data)
        if self.use_kernel:

            input_ = input
            input_ = input_ * (2 ** 16)
            input_ = input_.int()
            
            bias_ = self.bias
            bias_ = bias_ * (2 ** 16)
            bias_ = bias_.int()

            shift_ = self.shift
            shift_ = shift_.int()

            sign_ = self.sign
            sign_ = sign_.int()


        
        if not hasattr(self.shift,'org'):
            self.shift.org=self.shift.data.clone()
        self.shift.data=self.shift.org.round()

        if not hasattr(self.sign,'org'):
            self.sign.org=self.sign.data.clone()
        self.sign.data=self.sign.org.round()
        # print(input)
        # print(input_)
    
        if self.use_kernel:
            if(self.use_cuda):
                print("cuda kernel")
                nn = shift_cuda_kernel.linear_shift(input_, shift_, sign_, bias_)

                out = nn.float()

                out = out / (2**16)
                
                return out
            else:
                nn = shift_kernel.linear_kernel(input_.detach().numpy(), self.shift.detach().numpy(),self.sign.detach().numpy(),bias_.detach().numpy())

                out = torch.FloatTensor(nn)

                out = out / (2**16)
                return out


        else:
          
            weight = (2 ** self.shift) * ( (-1) ** self.sign )
            return F.linear(input, weight, self.bias)
       

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

# check gradient of linear_shift
linear_shift = LinearShift(20, 30, check_grad=True) # LinearShiftFunction.apply 

from torch.autograd import gradcheck
# gradcheck takes a tuple of tensors as input, check if your gradient
# evaluated with these tensors are close enough to numerical
# approximations and returns True if they all verify this condition.
data = torch.randn(20,20,dtype=torch.double,requires_grad=True)
weight = torch.randn(30,20,dtype=torch.double,requires_grad=True)
input = (data, weight)
# test = gradcheck(linear_shift, data, eps=1e-6, atol=1e-4)
# print("gradcheck result for linear_shift: ", test)


class _ConvNdShift(nn.Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias', 'padding_mode']

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode, check_grad=False):
        super(_ConvNdShift, self).__init__()
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

        if check_grad:
            tensor_constructor = torch.DoubleTensor # double precision required to check grad
        else:
            tensor_constructor = torch.Tensor # In PyTorch torch.Tensor is alias torch.FloatTensor

        if transposed:
            self.shift = nn.Parameter(tensor_constructor(
                in_channels, out_channels // groups, *kernel_size))
            self.sign = nn.Parameter(tensor_constructor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.shift = nn.Parameter(tensor_constructor(
                out_channels, in_channels // groups, *kernel_size))
            self.sign = nn.Parameter(tensor_constructor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(tensor_constructor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        weights = torch.zeros_like(self.shift)
        init.kaiming_uniform_(weights, a=math.sqrt(5))
        self.shift.data, self.sign.data = get_shift_and_sign(weights)

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.shift)
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

class Conv2dShift(_ConvNdShift):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', check_grad=False, use_kernel=False,use_cuda =True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.use_kernel = use_kernel
        self.use_cuda = use_cuda
        super(Conv2dShift, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

    #@weak_script_method
    def forward(self, input):
        input.data=round_to_fixed(input.data)
        self.bias.data=round_to_fixed(self.bias.data)
        if self.use_kernel:
            # start = time.time()

            input_ = input
            input_ = input_ * (2 ** 16)
            input_ = input_.int()
            
            bias_ = self.bias
            bias_ = bias_ * (2 ** 16)
            bias_ = bias_.int()
            shift_ = self.shift
            shift_ = shift_.int()

            sign_ = self.sign
            sign_ = sign_.int()

            # print("process data ", time.time() - start)

        if not hasattr(self.shift,'org'):
            self.shift.org=self.shift.data.clone()
        self.shift.data=self.shift.org.round()

        if not hasattr(self.sign,'org'):
            self.sign.org=self.sign.data.clone()
        self.sign.data=self.sign.org.round()
       

        if self.use_kernel:
            if(self.use_cuda):
                print("cuda kernel")
                input_ = F.pad(input = input_, pad = self.padding, mode = 'constant', value = 0)
                nn = shift_cuda_kernel.conv2d_shift(input_, shift_, sign_, bias_, self.stride, self.padding )
                out = nn.float()
                out = out / (2**16)
                return out
            else:
                input_ = F.pad(input = input_, pad = self.padding, mode = 'constant', value = 0)
                nn = shift_kernel.convolution_kernel(input_.detach().numpy(), 
                    self.shift.detach().numpy(),
                    self.sign.detach().numpy(),
                    bias_.detach().numpy(),self.stride, self.padding)
                out = torch.FloatTensor(nn)
                out = out / (2**16)
                return out

        else:
            weight = (2 ** self.shift) * ( (-1) ** self.sign )
            
            if self.padding_mode == 'circular':
                expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                    (self.padding[0] + 1) // 2, self.padding[0] // 2)
                return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                                weight, self.bias, self.stride,
                                _pair(0), self.dilation, self.groups)
            return F.conv2d(input, weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        
