import torch
import time
import torch.nn.functional as F
try:
    import deepshift_cuda
    import deepshift_cpu
except:
    print("Unable to import CPU and/or CUDA bit-wise shift kernels")

def linear(input, shift, sign, bias=None, conc_weight=None, use_cuda=True):
    if(use_cuda):   
        assert(conc_weight is not None)
        # start_time = time.time()      
        out = torch.zeros([input.size(0), shift.size(0)], dtype=torch.int32, device=input.device)
        if bias is not None:
            deepshift_cuda.DEEP_SHIFT_LINEAR(input, conc_weight.data, bias, out, conc_weight.base, conc_weight.bits, shift.size(0))
        else:
            temp = torch.zeros([shift.size(0)], dtype=torch.int32, device=input.device)
            deepshift_cuda.DEEP_SHIFT_LINEAR(input, conc_weight.data, temp, out, conc_weight.base, conc_weight.bits, shift.size(0))
        # end_time = time.time()
        # print("Linear Time:", end_time - start_time )    
    else:
        out = deepshift_cpu.linear_kernel(input.detach().numpy(), shift.detach().numpy(), sign.detach().numpy(), bias.detach().numpy())
        out = torch.Tensor(out)

    return out

def conv2d(input, shift, sign, bias=None, conc_weight=None, stride=1, padding=0, dilation=1, groups=1, use_cuda=True):
    if(use_cuda):
        assert(conc_weight is not None)
        start_time = time.time()
        if len(padding) == 2:
            padding = (padding[0], padding[0], padding[1], padding[1])
        else:
            padding = padding
        input = F.pad(input = input, pad = padding, mode = 'constant', value = 0)
        if len(stride) == 1:
            strides_h = stride[0]
            strides_w = stride[0]
        else: 
            strides_h = stride[0]
            strides_w = stride[1]
        kernel_size = shift.shape[2:4]
        out_height = int((input.size(2) - kernel_size[0]) / strides_h +1)
        out_width = int((input.size(3) - kernel_size[1]) / strides_w +1)
        out_channels = shift.size(0)
        out = torch.zeros([input.size(0), out_channels, out_height, out_width], dtype=torch.int32, device=input.device)

        if bias is not None:
            deepshift_cuda.DEEP_SHIFT_CONV(input, conc_weight.data, bias, out, stride, padding, kernel_size[0], kernel_size[1], conc_weight.base, conc_weight.bits)
        else:
            temp = torch.zeros([out_channels], dtype=torch.int32, device=input.device)
            deepshift_cuda.DEEP_SHIFT_CONV(input, conc_weight.data, temp, out, stride, padding, kernel_size[0], kernel_size[1], conc_weight.base, conc_weight.bits)
        # end_time = time.time()
        # print("Conv Time:", end_time - start_time )
    
    else:
        input = F.pad(input = input, pad = padding, mode = 'constant', value = 0)
        out = deepshift_cpu.convolution_kernel(input.cpu().detach().numpy(), 
                                                  shift.cpu().detach().numpy(),
                                                  sign.cpu().detach().numpy(),
                                                  bias.cpu().detach().numpy(), stride, padding)
        out = torch.Tensor(out)

    #print("out - out1: ", out.cpu() - out1.cpu().int())

    return out

def compress_sign_and_shift(shift, sign, comp_size, base, bits, row_length, num):
    comp_weight = torch.zeros([comp_size], dtype=torch.int32,device = torch.device('cuda:0'))
    
    deepshift_cuda.COMPRESS_SIGN_SHIFT(shift, sign, comp_weight, base, bits, shift.shape[0], shift.shape[1], shift.shape[2], shift.shape[3], row_length, num)

    return comp_weight