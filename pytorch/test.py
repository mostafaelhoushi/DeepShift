import torch
import torchvision

N, C, H, W = 256, 3, 224, 224 #256, 3, 224, 224
k, c, h, w = 3, 3, 7, 7 #64, 3, 7, 7
assert c==C

X = torch.rand((N, C, H, W))
A = torch.rand((k, c, h, w))
B = torch.rand((k, c, h, w))

conv = torch.nn.Conv2d(c, k, (h,w), stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros')
conv.weight.data = A+B
Y = conv(X)

conv1 = torch.nn.Conv2d(c, k, (h,w), stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros')
conv2 = torch.nn.Conv2d(c, k, (h,w), stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros')
conv1.weight.data = A
conv2.weight.data = B
Y_new = conv1(X) + conv2(X)

MSE = torch.sum((Y - Y_new)**2)
print(MSE.detach().numpy())