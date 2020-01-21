import torch
import torch.nn as nn
import numpy as np

from .unoptimized_linear import UnoptimizedLinear
from .unoptimized_conv import UnoptimizedConv2d

def convert_to_unoptimized(model):
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = convert_to_unoptimized(model=module)
        if type(module) == nn.Linear:
            linear = module       
            unoptimized_linear = UnoptimizedLinear(module.in_features, module.out_features, module.bias is not None) 
            unoptimized_linear.weight = linear.weight
            unoptimized_linear.bias = linear.bias

            model._modules[name] = unoptimized_linear
        if type(module) == nn.Conv2d:
            conv2d = module
            unoptimized_conv = UnoptimizedConv2d(module.in_channels, module.out_channels, module.kernel_size, module.stride,
                                                 module.padding, module.dilation, module.groups,
                                                 module.bias is not None, module.padding_mode) 
            unoptimized_conv.bias = conv2d.bias
            unoptimized_conv.weight = conv2d.weight

            model._modules[name] = unoptimized_conv

    return model
    
    
if __name__ == '__main__':
    # this test will be run if you type in the command:
    # > python convert_to_unoptimized    
    import torchvision.models as models
    model = models.__dict__['resnet18'](pretrained=True)
    model = model.to("cuda:0")
    input = torch.rand((32, 3, 224, 224)).to("cuda:0")
    output1 = model(input)


    model = convert_to_unoptimized(model).to("cuda:0")
    output2 = model(input)

    max_error = torch.max(torch.abs(output1 - output2))
    print(max_error.detach().cpu().numpy())

