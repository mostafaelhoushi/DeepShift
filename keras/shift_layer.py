from keras import backend as K
from keras.layers import Layer
import tensorflow as tf
import numpy as np
import math
import sys
sys.path.insert(0, '../spfpm/')
from FixedPoint import FXfamily, FXnum
import keras.constraints
from keras.initializers import RandomUniform
from keras.constraints import Constraint
from keras.regularizers import L1L2

from tensorflow.python.framework import tensor_shape
import time

class IntegerConstraint (Constraint):
    def __init__(self, low=None, high=None, **kwargs):
        super(IntegerConstraint, self).__init__(**kwargs)
        self.low = low
        self.high = high

    def __call__(self, w):
        #print("w: " + str(w.numpy()))
        res = K.round(w)

        if self.low is not None and self.high is not None:
            res = K.clip(res, self.low, self.high)
        elif self.low is not None and self.high is None:
            res = K.clip(res, self.low, res)
        elif self.high is not None and self.low is None:
            res = K.clip(res, res, self.high)

        return res

class ConvertToSignConstraint (Constraint):
    def __init__(self, **kwargs):
        super(ConvertToSignConstraint, self).__init__(**kwargs)

    def __call__(self, w):
        return K.cast(K.greater_equal(w, 0), dtype='float32')*(+1) + K.cast(K.less(w,0), dtype='float32')*(-1)    

class RoundedRandomUniform(RandomUniform):
    def __init__(self, minval=-10, maxval=-1, seed=None):
        super(RoundedRandomUniform, self).__init__(minval, maxval, seed)

    def __call__(self, shape, dtype=None):
        return K.round(super(RoundedRandomUniform, self).__call__(shape, dtype))

class L1L2_PowerOf2(L1L2):
    """Regularizer for L1 and L2 regularization for shift weights.
    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    """
    def __call__(self, x):
        twos = K.ones(shape=x.shape)*2
        twos_power_of_x = K.pow(twos, x)
        regularization = 0
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
        self.shift = self.add_weight(name='shift',
                                      shape=(input_shape[1], self.output_dim),
                                      constraint=IntegerConstraint(),
                                      #dtype=tf.int32,
                                      initializer=RoundedRandomUniform(),
                                      trainable=True)
        self.sign = self.add_weight(name='sign',
                                      shape=(input_shape[1], self.output_dim),
                                      constraint=IntegerConstraint(0,1),
                                      #dtype=tf.int32,
                                      initializer=RoundedRandomUniform(0,1),
                                      trainable=True)
        self.bias = self.add_weight(name='bias',
                                      shape=(1, self.output_dim,),
                                      initializer='uniform',
                                      trainable=True)

        self.twos = K.ones(shape=self.shift.shape)*2
        self.minusones = K.ones(shape=self.sign.shape)*-1

        super(DenseShift, self).build(input_shape)

    def inference_fun(self, x):

        x_numpy = x.numpy()
        shift_numpy = self.shift.numpy()
        sign_numpy = self.sign.numpy()

        x_result = np.zeros((x_numpy.shape[0], self.output_dim))

        # TODO: handle arbitrary dimensions
        for i in range(x_numpy.shape[0]):
            for j in range(self.output_dim):
                for k in range(0, x_numpy.shape[1], 3):
                    val = x_numpy[i][k]
                    s = sign_numpy[k][j]
                    sft = shift_numpy[k][j]
                    if (val != 0):
                        val_fp = FXnum(val)

                        if (sft > 0):
                            val_fp = val_fp << int(abs(sft))
                        else:
                            val_fp = val_fp >> int(abs(sft))
                        if s == 1:
                            val_fp = -val_fp
                        x_result[i][j] += float(val_fp)

                    #for k+1
                    if (k+1 < x_numpy.shape[1]):
                        val = x_numpy[i][k+1]
                        s = sign_numpy[k+1][j]
                        sft = shift_numpy[k+1][j]
                        if (val != 0):
                            val_fp = FXnum(val)

                            if (sft > 0):
                                val_fp = val_fp << int(abs(sft))
                            else:
                                val_fp = val_fp >> int(abs(sft))
                            if s == 1:
                                val_fp = -val_fp
                            x_result[i][j] += float(val_fp)

                    #for k+2
                    if (k+2 < x_numpy.shape[1]):
                        val = x_numpy[i][k+2]
                        s = sign_numpy[k+2][j]
                        sft = shift_numpy[k+2][j]
                        if (val != 0):
                            val_fp = FXnum(val)

                            if (sft > 0):
                                val_fp = val_fp << int(abs(sft))
                            else:
                                val_fp = val_fp >> int(abs(sft))
                            if s == 1:
                                val_fp = -val_fp
                            x_result[i][j] += float(val_fp)


        x_result = tf.convert_to_tensor(x_result, dtype=np.float32)

        x_result += self.bias

        return x_result

    def call(self, x):
        W = K.pow(self.twos, self.shift) * K.pow(self.minusones, self.sign)
        return K.dot(x,W) + self.bias
        '''
        if K.in_train_phase(True, False):
            W = K.pow(self.twos, self.shift) * K.pow(self.minusones, self.sign)
            return K.dot(x,W) + self.bias
        else:
            return self.inference_fun(x)
        '''



    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

