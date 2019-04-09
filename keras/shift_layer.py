from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import tensorflow as tf
import numpy as np

import tensorflow.keras.constraints
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.regularizers import L1L2

class IntegerConstraint (Constraint):
    def __init__(self, low=None, high=None, **kwargs):
        super(IntegerConstraint, self).__init__(**kwargs)
        self.low = low
        self.high = high

    def __call__(self, w):
        res = K.round(w)
        if self.low is not None:
            res = res*K.cast(K.greater_equal(res, self.low), K.floatx()) + self.low*K.cast(K.less_equal(res, self.low), K.floatx())
        if self.high is not None:
            res = res*K.cast(K.less_equal(res, self.high), K.floatx()) + self.high*K.cast(K.greater_equal(res, self.high), K.floatx())

        return res

class RoundedRandomUniform(RandomUniform):
    def __init__(self, minval=-10, maxval=-1, seed=None):
        super(RoundedRandomUniform, self).__init__(minval, maxval, seed)

    def __call__(self, shape, dtype=None, partition_info=None):
        return K.round(super(RoundedRandomUniform, self).__call__(shape, dtype, partition_info))

class L1L2_PowerOf2(L1L2):
    """Regularizer for L1 and L2 regularization for shift weights.
    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    """
    def __call__(self, x):
        twos = K.ones(shape=x.shape)*2
        twos_power_of_x = K.pow(twos, x)
        regularization = 0.
        if self.l1:
            regularization += K.sum(self.l1 * K.abs(twos_power_of_x))
        if self.l2:
            regularization += K.sum(self.l2 * K.square(twos_power_of_x))
        return regularization

def l1_powerof2(l=0.01):
    return L1L2_PowerOf2(l1=l)


def l2_powerof2(l=0.01):
    return L1L2_PowerOf2(l2=l)


def l1_l2_powerof2(l1=0.01, l2=0.01):
    return L1L2_PowerOf2(l1=l1, l2=l2)

class DenseShift(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(DenseShift, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.shift = self.add_variable(name='shift', 
                                      shape=[input_shape[1], self.output_dim],
                                      constraint=IntegerConstraint(),
                                      #dtype=tf.int32,
                                      initializer=RoundedRandomUniform(),
                                      trainable=True)
        self.sign = self.add_variable(name='sign', 
                                      shape=[input_shape[1], self.output_dim],
                                      constraint=IntegerConstraint(0,1),
                                      #dtype=tf.int32,
                                      initializer=RoundedRandomUniform(0,1),
                                      trainable=True)                        
        self.bias = self.add_variable(name='bias', 
                                    shape=[1, self.output_dim],
                                    initializer='uniform',
                                    trainable=True)

        self.twos = K.ones(shape=self.shift.shape)*2
        self.minusones = K.ones(shape=self.sign.shape)*-1

        super(DenseShift, self).build(input_shape) 

    def call(self, x):
        W = K.pow(self.twos, self.shift) * K.pow(self.minusones, self.sign)
        return K.dot(x,W) + self.bias

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

