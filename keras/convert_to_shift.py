import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, DepthwiseConv2D
from tensorflow.keras.models import Model

from shift_layer import DenseShift
from convolutional_shift import Conv2DShift, DepthwiseConv2DShift
from shift_utils import * 

def convert_to_shift(model, num_layers = -1, num_to_replace=None, convert_weights=False, freeze=False):
    # create input layer for new model
    input_shape = model.input.shape[1:] # first shape element is batch size so don't copy it
    inputs = tf.keras.Input(shape=input_shape)

    # copy the layers - except the input layer - from original model
    layers = [l for l in model.layers[1:]]

    # count number of instances to keep
    num_layer_type = sum(1 for l in layers if type(l)==Conv2D or type(l)==DepthwiseConv2D)
    if (num_to_replace is not None):
        num_layer_keep = max(num_layer_type - num_to_replace, 0)
    else:
        num_layer_keep = num_layer_type

    num_layer_kept = 0
    x = inputs
    for i, layer in enumerate(layers):
        if type(layer) == Dense and num_to_replace is not None:
            input = layer.input
            output = layer.output
            weights = layer.weights

            # weights of Dense has shape: (features_in, features_out)
            _, features_out = weights[0].shape
            #TODO: copy all other attributes. Consider using get_config() and from_config()
            dense_shift_layer = DenseShift(weights[0].shape[-1]) 

            x = dense_shift_layer(x)

            if (convert_weights is True):
                [kernel, bias] = weights

                shift, sign = get_shift_and_sign(kernel)
                # convert sign to (-1)^sign
                # i.e., 1 -> 0, -1 -> 1
                sign = sign.numpy()
                sign[sign == 1] = 0
                sign[sign == -1] = 1
                
                new_weights = [shift, sign, bias]

                dense_shift_layer.set_weights(new_weights)

        elif type(layer) == Conv2D:
            num_layer_kept += 1
            if num_layer_kept > num_layer_keep: 
                input = layer.input
                output = layer.output
                weights = layer.weights

                # weights of Conv2D has shape: (filter_height, filter_width, channels_in, channels_out)
                filter_height, filter_width, _, channels_out = weights[0].shape.as_list()
                #TODO: copy all other attributes. Consider using get_config() and from_config()
                conv2d_shift_layer = Conv2DShift(filters=channels_out, kernel_size = (filter_height, filter_width)) 

                x = conv2d_shift_layer(x)

                if (convert_weights is True):
                    if layer.use_bias:
                        [kernel, bias] = layer.get_weights()
                    else:
                        [kernel] = layer.get_weights()
                        bias = np.zeros(shape=(channels_out,))

                    #TODO: if attributes copied, then need to check has_bias also on conv2d_shift_layer

                    shift, sign = get_shift_and_sign(kernel)
                    # convert sign to (-1)^sign
                    # i.e., 1 -> 0, -1 -> 1
                    sign = sign.numpy()
                    sign[sign == 1] = 0
                    sign[sign == -1] = 1

                    new_weights = [shift, bias, sign]

                    conv2d_shift_layer.set_weights(new_weights)
                
            else:
                x = layer(x)
                if freeze:
                    layer.trainable = False

        elif type(layer) == DepthwiseConv2D:
            num_layer_kept += 1
            if num_layer_kept > num_layer_keep: 
                config = layer.get_config()
                config.pop("name")
                depthwise_conv2d_shift_layer = DepthwiseConv2DShift.from_config(config)

                x = depthwise_conv2d_shift_layer(x)

                if (convert_weights is True):
                    if layer.use_bias:
                        [kernel, bias] = layer.get_weights()
                    else:
                        [kernel] = layer.get_weights()

                    #TODO: if attributes copied, then need to check has_bias also on conv2d_shift_layer

                    shift, sign = get_shift_and_sign(kernel)
                    # convert sign to (-1)^sign
                    # i.e., 1 -> 0, -1 -> 1
                    sign = sign.numpy()
                    sign[sign == 1] = 0
                    sign[sign == -1] = 1

                    if layer.use_bias:
                        new_weights = [shift, bias, sign]
                    else:
                        new_weights = [shift, sign]
                    
                    depthwise_conv2d_shift_layer.set_weights(new_weights)
            else:
                x = layer(x)
                if freeze:
                    layer.trainable = False

        else:
            x = layer(x)
            if num_layer_kept <= num_layer_keep: 
                if freeze:
                    layer.trainable = False
    outputs = x

    model_converted = Model(inputs=inputs, outputs=outputs)


    #TODO: Copy other attributes such as learning rate, optimizer, etc.
    return model_converted

'''
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.keras.models import Model

from shift_layer import DenseShift
from convolutional_shift import Conv2DShift

# Source: https://stackoverflow.com/a/54517478/3880948
def insert_layer(model, layer_type, insert_layer_factory,
                 insert_layer_name=None, position='replace', num_to_replace=None):
    # copy the layers
    layers = [l for l in model.layers]

    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in layers:
        for node in layer.outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                        {layer_name: [layer.name]})
            else:
                if (layer.name not in network_dict['input_layers_of'][layer_name]):
                    network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
            {layers[0].name: layers[0].input})

    # count number of instances to keep
    num_layer_type = sum(1 for l in layers if type(l)==layer_type)
    if (num_to_replace is not None):
        num_layer_keep = max(num_layer_type - num_to_replace, 0)
    else:
        num_layer_keep = num_layer_type

    num_layer_kept = 0
    # Iterate over all layers after the input
    for layer in layers[1:]:
        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux] 
                for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Insert layer if name matches the regular expression
        if type(layer) == layer_type:
            num_layer_kept += 1
            if num_layer_kept > num_layer_keep: 
                if position == 'replace':
                    x = layer_input
                elif position == 'after':
                    x = layer(layer_input)
                elif position == 'before':
                    pass
                else:
                    raise ValueError('position must be: before, after or replace')

                new_layer = insert_layer_factory(layer)
                if insert_layer_name:
                    new_layer.name = insert_layer_name
                
                x = new_layer(x)
                print('Layer {} inserted after layer {}'.format(new_layer.name,
                                                                layer.name))
                if position == 'before':
                    x = layer(x)
            else:
                x = layer(layer_input)
        else:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

    return Model(inputs=layers[0].input, outputs=x)

def convert_to_conv2d_shift(conv2d_layer):
    if conv2d_layer.use_bias:
        [weights, bias] = conv2d_layer.weights
    else:
        [weights] = conv2d_layer.weights

    # weights of Conv2D has shape: (filter_height, filter_width, channels_in, channels_out)
    filter_height, filter_width, _, channels_out = weights.shape.as_list()
    #TODO: copy all other attributes. Consider using get_config() and from_config()
    return Conv2DShift(filters=channels_out, kernel_size = (filter_height, filter_width)) 

def convert_to_dense_shift(dense_layer):
    [weights, bias] = dense_layer.weights

    # weights of Dense has shape: (features_in, features_out)
    _, features_out = weights.shape
    #TODO: copy all other attributes. Consider using get_config() and from_config()
    return DenseShift(features_out) 

def convert_to_shift(model, num_to_replace = None):
    model_converted = insert_layer(model, Dense, convert_to_dense_shift, num_to_replace = 1)
    model_converted.summary()
    model_converted = insert_layer(model_converted, Conv2D, convert_to_conv2d_shift, num_to_replace = num_to_replace)

    #TODO: Copy other attributes such as learning rate, optimizer, etc.
    return model_converted
'''
