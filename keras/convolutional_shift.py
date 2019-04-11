import tensorflow as tf
from tensorflow.python.keras.layers.convolutional import *

from shift_layer import *

class ConvShift(Conv):
  def __init__(self, rank,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               data_format=None,
               dilation_rate=1,
               activation=None,
               use_bias=True,
               kernel_initializer=RoundedRandomUniform(),
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=IntegerConstraint(),
               bias_constraint=None,
               trainable=True,
               name=None,
               **kwargs):
    super(ConvShift, self).__init__(rank=rank,
                                    filters=filters,
                                    kernel_size=kernel_size,
                                    strides=strides,
                                    padding=padding,
                                    data_format=data_format,
                                    dilation_rate=dilation_rate,
                                    activation=activation,
                                    use_bias=use_bias,
                                    kernel_initializer=kernel_initializer,
                                    bias_initializer=bias_initializer,
                                    kernel_regularizer=kernel_regularizer,
                                    bias_regularizer=bias_regularizer,
                                    activity_regularizer=activity_regularizer,
                                    kernel_constraint=kernel_constraint,
                                    bias_constraint=bias_constraint,
                                    trainable=trainable,
                                    name=name,
                                    **kwargs)

  def build(self, input_shape):
    # TODO: minimize redundancy by invoking build of Conv
    # and do minimal additions here
    super(ConvShift, self).build(input_shape)

    kernel_shape = self.kernel.shape

    #TODO: rename "self.kernel" to "self.shift" ?
    self.sign = self.add_weight(shape=kernel_shape,
                                    initializer=RoundedRandomUniform(0,1),
                                    name='sign',
                                    regularizer=None,
                                    constraint=IntegerConstraint(0,1),
                                    trainable=True)    

    self.twos = K.ones(shape=self.kernel.shape)*2
    self.minusones = K.ones(shape=self.sign.shape)*-1


  def call(self, inputs):
    W = K.pow(self.twos, self.kernel) * K.pow(self.minusones, self.sign)
    outputs = self._convolution_op(inputs, W)

    if self.use_bias:
      if self.data_format == 'channels_first':
        if self.rank == 1:
          # nn.bias_add does not accept a 1D input tensor.
          bias = array_ops.reshape(self.bias, (1, self.filters, 1))
          outputs += bias
        else:
          outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
      else:
        outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def compute_output_shape(self, input_shape):
    return super(ConvShift, self).compute_output_shape(input_shape)

  def get_config(self):
    config = {
        'filters': self.filters,
        'kernel_size': self.kernel_size,
        'strides': self.strides,
        'padding': self.padding,
        'data_format': self.data_format,
        'dilation_rate': self.dilation_rate,
        'activation': activations.serialize(self.activation),
        'use_bias': self.use_bias,
        'kernel_initializer': initializers.serialize(self.kernel_initializer),
        'bias_initializer': initializers.serialize(self.bias_initializer),
        'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint': constraints.serialize(self.kernel_constraint),
        'bias_constraint': constraints.serialize(self.bias_constraint)
    }
    base_config = super(ConvShift, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def _compute_causal_padding(self):
    """Calculates padding for 'causal' option for 1-d conv layers."""
    return super(ConvShift, self)._compute_causal_padding()
    

class Conv2DShift(ConvShift):
  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
               activation=None,
               use_bias=True,
               kernel_initializer=RoundedRandomUniform(),
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=IntegerConstraint(),
               bias_constraint=None,
               **kwargs):
    super(Conv2DShift, self).__init__(
        rank=2,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activations.get(activation),
        use_bias=use_bias,
        kernel_initializer=initializers.get(kernel_initializer),
        bias_initializer=initializers.get(bias_initializer),
        kernel_regularizer=regularizers.get(kernel_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        kernel_constraint=constraints.get(kernel_constraint),
        bias_constraint=constraints.get(bias_constraint),
        **kwargs)

    def inference_fun(self, x):
        # TODO: implement bitwise
        return super(Conv2DShift, self).call(x)

    def call(self, inputs):
        if K.in_train_phase(True, False):
            return super(Conv2DShift, self).call(x)
        else:
            return self.inference_fun(x)

# Aliases

Convolution2DShift = Conv2DShift
