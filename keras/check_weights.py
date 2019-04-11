from keras.layers import DepthwiseConv2D, Flatten, Softmax, Conv2D
from keras.models import Model, load_model
from keras.datasets import cifar10
from keras import optimizers
import keras.applications

import numpy as np

from shift_layer import *
from convolutional_shift import *

from cifar10_resshift import *

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

#model = load_model("./saved_models.shift_last_stack_only/cifar10_ResNet20v1_model.144.h5", custom_objects={'Conv2DShift': Conv2DShift, 'DenseShift': DenseShift})
model = resnet_v1(input_shape=(32,32,3), depth=20, shift_depth=56)
model.load_weights("./saved_models/cifar10_ResNet20v1_model_shift_22/model.003.h5")

for layer in model.layers:
  if isinstance(layer, Conv2DShift) or isinstance(layer, DenseShift):
    print(layer.name)
    print(len(layer.get_weights()))
    print(layer.get_weights())