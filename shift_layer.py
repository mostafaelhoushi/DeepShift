from keras import backend as K
from keras.layers import Layer

class DenseShift(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(DenseShift, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.shift = self.add_weight(name='shift', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        self.bias = self.add_weight(name='bias', 
                                    shape=(1, self.output_dim),
                                    initializer='uniform',
                                    trainable=True)
        super(DenseShift, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return K.dot(x, K.pow(2.0, self.shift)) + self.bias

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)