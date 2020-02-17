import torch
try:
    import unoptimized_cuda
except:
    print("Unable to import CUDA unoptimized kernels")

def linear(input, weight, bias):
    out = torch.zeros([input.size(0), weight.size(0)], dtype=torch.float, device=torch.device('cuda:0'))
    if bias is not None:
        unoptimized_cuda.UNOPTIMIZED_LINEAR(input, weight, bias, out)
    else:
        temp = torch.zeros([weight.size(0)], dtype=torch.float, device=torch.device('cuda:0'))
        unoptimized_cuda.UNOPTIMIZED_LINEAR(input, weight, temp, out)

    return out

def conv2d(input, weight, bias, stride, padding):
    if len(stride) == 1:
        strides_h = stride[0]
        strides_w = stride[0]
    else: 
        strides_h = stride[0]
        strides_w = stride[1]
    out_height = int((input.size(2) - weight.size(2)) / strides_h +1)
    out_width = int((input.size(3) - weight.size(3)) / strides_w +1)
    out = torch.zeros([input.size(0), weight.size(0), out_height, out_width], dtype=torch.float, device=torch.device('cuda:0'))
    
    if bias is not None:
        unoptimized_cuda.UNOPTIMIZED_CONV(input, weight, bias, out, stride, padding )
    else:
        temp = torch.zeros([weight.size(0)], dtype=torch.float, device=torch.device('cuda:0'))
        unoptimized_cuda.UNOPTIMIZED_CONV(input, weight, temp, out, stride, padding )

    return out