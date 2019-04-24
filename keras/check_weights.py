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

model_dir = "./selected_models/cifar10/ResNet50v1/model_shift_20/"
model_file = "model.040.h5"

model = resnet_v1(input_shape=(32,32,3), depth=50, shift_depth=20)
model.load_weights(os.path.join(model_dir, model_file))

count = 0
for layer in model.layers:
  if isinstance(layer, Conv2DShift) or isinstance(layer, DenseShift):
    print(layer.name)
    print(len(layer.get_weights()))
    print(layer.get_weights())

    for index, w in enumerate(layer.get_weights()):
      weights_csv_name = layer.name + "_" + str(index) + ".csv"
      if len(w.shape) > 2:
        w = np.reshape(np.transpose(w, (1,0,2,3)), (w.shape[0],-1))
      np.savetxt(os.path.join(model_dir, weights_csv_name), w, fmt="%1.4f", delimiter=",")
