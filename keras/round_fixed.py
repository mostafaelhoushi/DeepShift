import sys
######necessary packages to use spfpm, the fixed point package
sys.path.insert(0, '../spfpm/')
from FixedPoint import FXfamily, FXnum
######

from keras import backend as K
from keras.layers import Layer

class RoundToFixed(Layer):

    def __init__(self, **kwargs):
        super(RoundToFixed, self).__init__(**kwargs)

    def build(self, input_shape):
        # This layer has no trainable weights
        super(RoundToFixed, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return FXnum(x)

    def compute_output_shape(self, input_shape):
        # this is an elementwise op so output shape is same as input shape
        return input_shape