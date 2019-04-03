from keras import backend as K
from keras.layers import Layer
import tensorflow as tf
import numpy as np

import keras.constraints
from keras.constraints import Constraint

class IntegerConstraint (Constraint):
    def __call__(self, w):
        return K.round(w)

def randpsuedoint(shape):
    return K.round(K.random_uniform_variable(shape, low=-10, high=-1)) # K.round(K.random_normal(shape, dtype))

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
                                      initializer=randpsuedoint,
                                      trainable=True)
        self.bias = self.add_weight(name='bias', 
                                    shape=(1, self.output_dim),
                                    initializer='uniform',
                                    trainable=True)

        self.twos = K.ones(shape=self.shift.shape)*2

        super(DenseShift, self).build(input_shape) 

    def call(self, x):
        W = K.pow(self.twos, self.shift)
        return K.dot(x,W) + self.bias

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

