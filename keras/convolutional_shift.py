from keras.layers.convolutional import _Conv
from keras.legacy import interfaces
from keras.engine.base_layer import InputSpec
from shift_layer import *

class _ConvShift(_Conv):
    # TODO: Handle if different initializer or constraint is passed
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
                 **kwargs):
        super(_ConvShift, self).__init__(rank=rank,
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
                                         **kwargs)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)


        # Overwrite kernel to be shifts only
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=RoundedRandomUniform(), # self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=IntegerConstraint()) #self.kernel_constraint)

        self.sign = self.add_weight(shape=kernel_shape,
                                      initializer=RoundedRandomUniform(0,1),
                                      name='sign',
                                      regularizer=None,
                                      constraint=IntegerConstraint(0,1),
                                      trainable=True)    

        self.twos = K.ones(shape=self.kernel.shape)*2
        self.minusones = K.ones(shape=self.sign.shape)*-1

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    # Overwrite call to perform bit shifts instead of multiplcations
    def call(self, inputs):
        W = K.pow(self.twos, self.kernel) * K.pow(self.minusones, self.sign)

        if self.rank == 1:
            outputs = K.conv1d(
                inputs,
                W,
                strides=self.strides[0],
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate[0])
        if self.rank == 2:
            outputs = K.conv2d(
                inputs,
                W,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)
        if self.rank == 3:
            outputs = K.conv3d(
                inputs,
                W,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

class Conv2DShift(_ConvShift):
    @interfaces.legacy_conv2d_support
    def __init__(self, filters,
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
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

    def get_config(self):
        config = super(Conv2DShift, self).get_config()
        config.pop('rank')
        return config

class DepthwiseConv2DShift(Conv2DShift):
    def __init__(self,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               depth_multiplier=1,
               data_format=None,
               activation=None,
               use_bias=True,
               depthwise_initializer=RoundedRandomUniform(),
               bias_initializer='zeros',
               depthwise_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               depthwise_constraint=IntegerConstraint(),
               bias_constraint=None,
               **kwargs):
        super(DepthwiseConv2DShift, self).__init__(
            filters=None,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            bias_constraint=bias_constraint,
            **kwargs)
        #TODO: Better handling of overriding initializer and constraint
        self.depth_multiplier = depth_multiplier
        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
        self.depthwise_constraint = constraints.get(depthwise_constraint)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        if len(input_shape) < 4:
            raise ValueError('Inputs to `DepthwiseConv2D` should have rank 4. '
                            'Received input shape:', str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs to '
                                             '`DepthwiseConv2D` '
                                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        depthwise_kernel_shape = (self.kernel_size[0],
                                    self.kernel_size[1],
                                    input_dim,
                                    self.depth_multiplier)

        self.depthwise_kernel = self.add_weight(
                shape=depthwise_kernel_shape,
                initializer=RoundedRandomUniform(), # self.depthwise_initializer,
                name='depthwise_kernel',
                regularizer=self.depthwise_regularizer,
                constraint=IntegerConstraint()) # self.depthwise_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(input_dim * self.depth_multiplier,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        # Set input spec.
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

        self.sign = self.add_weight(shape=depthwise_kernel_shape,
                                    initializer=RoundedRandomUniform(0,1),
                                    name='sign',
                                    regularizer=None,
                                    constraint=IntegerConstraint(0,1),
                                    trainable=True)        

        self.twos = K.ones(shape=self.depthwise_kernel.shape)*2
        self.minusones = K.ones(shape=self.sign.shape)*-1

    def call(self, inputs):
        W = K.pow(self.twos, self.depthwise_kernel) * K.pow(self.minusones, self.sign)
        outputs = backend.depthwise_conv2d(
                inputs,
                W,
                strides=self.strides,
                padding=self.padding,
                dilation_rate=self.dilation_rate,
                data_format=self.data_format)

        if self.use_bias:
            outputs = backend.bias_add(
                    outputs,
                    self.bias,
                    data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
            out_filters = input_shape[1] * self.depth_multiplier
        elif self.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]
            out_filters = input_shape[3] * self.depth_multiplier

        rows = conv_utils.conv_output_length(rows, self.kernel_size[0],
                                                   self.padding,
                                                   self.strides[0])
        cols = conv_utils.conv_output_length(cols, self.kernel_size[1],
                                                   self.padding,
                                                   self.strides[1])
        if self.data_format == 'channels_first':
            return (input_shape[0], out_filters, rows, cols)
        elif self.data_format == 'channels_last':
            return (input_shape[0], rows, cols, out_filters)

    def get_config(self):
        config = super(DepthwiseConv2DShift, self).get_config()
        config.pop('filters')
        config.pop('kernel_initializer')
        config.pop('kernel_regularizer')
        config.pop('kernel_constraint')
        config['depth_multiplier'] = self.depth_multiplier
        config['depthwise_initializer'] = initializers.serialize(
                self.depthwise_initializer)
        config['depthwise_regularizer'] = regularizers.serialize(
                self.depthwise_regularizer)
        config['depthwise_constraint'] = constraints.serialize(
                self.depthwise_constraint)
        return config

# Aliases
Convolution2DShift = Conv2DShift
