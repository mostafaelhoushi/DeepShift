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