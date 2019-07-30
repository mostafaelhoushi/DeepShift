'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202', 'vgg11', 'vgg13', 'vgg16']

def _weights_init(m):
    classname = m.__class__.__name__
    print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])

def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])

def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])

def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])

def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])

def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])

def test_resnet(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))

# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 09:39:30 2018

@author: tshzzz
"""

cfg = {
       'vgg11':[64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M'],
       'vgg13':[64,64,'M',128,128,'M',256,256,'M',512,512,'M',512,512,'M'],
       'vgg16':[64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M']
       
       }

class VGG(nn.Module):
    
    def __init__(self,layers,num_class=10):
        super(VGG,self).__init__()
        
        self.img_channel = 3
        self.layers = layers
        
        self.conv = self.make_layers()
        self.fc = nn.Linear(512, num_class)
        
    def make_layers(self):
        
        layers = []
        in_channel = self.img_channel
        for idx in range(len(self.layers)):
            if self.layers[idx] == 'M':          
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channel,self.layers[idx],kernel_size=3,padding=1))
                layers.append(nn.BatchNorm2d(self.layers[idx]))
                layers.append(nn.ReLU())
                in_channel = self.layers[idx]
        return nn.Sequential(*layers)
            
    
    def forward(self,x):
        
        out = self.conv(x)
        out = out.view(-1,512)
        out = self.fc(out)
        
        return out
        

def vgg11():
    return VGG(cfg['vgg11'],num_class=10)

def vgg13():
    return VGG(cfg['vgg13'],num_class=10)
       
def vgg16():
    return VGG(cfg['vgg16'],num_class=10)

def test_vgg():
    import torchvision.models as models
    model = models.vgg16()
    count = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            count += 1
    print(count)
    a = VGG16()
    count = 0
    for m in a.modules():
        if isinstance(m, nn.Conv2d):
            count += 1
    print(count)
    y = a(torch.randn(3,3,32,32))
    print(y.size())

if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test_resnet(globals()[net_name]())
            print()
    
    test_vgg()

