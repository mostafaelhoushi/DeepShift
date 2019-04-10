import sys
######necessary packages to use spfpm, the fixed point package
sys.path.insert(0, '../spfpm/')
from FixedPoint import FXfamily, FXnum
######

import tensorflow as tf
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

class RoundToFixed(Layer):

    def __init__(self, **kwargs):
        super(RoundToFixed, self).__init__(**kwargs)

    def build(self, input_shape):
        # This layer has no trainable weights
        super(RoundToFixed, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        # using the statement below made the train hang for some reason
        # return K.map_fn(lambda i : K.map_fn(lambda j : K.variable(float(FXnum(np.asscalar(j.numpy())))), i), x)
        #print("x: " + str(x))
        x_numpy = x.numpy()
        x_rounded = np.zeros(x_numpy.shape)
        
        fam = FXfamily(64)
        # TODO: handle arbitrary dimensions
        for i in range(x_numpy.shape[0]):
            for j in range(x_numpy.shape[1]):
                val = x_numpy[i][j]
                x_rounded[i][j] = float(FXnum(np.asscalar(val)))

        x_rounded = tf.convert_to_tensor(x_rounded, dtype=tf.float32)
        #print("x_rounded: " + str(x_rounded))
        return x_rounded


    def compute_output_shape(self, input_shape):
        # this is an elementwise op so output shape is same as input shape
        return input_shape