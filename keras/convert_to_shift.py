import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.keras.models import Model

from shift_layer import DenseShift
from convolutional_shift import Conv2DShift

def convert_to_shift(model, num_layers = -1):
    # create input layer for new model
    input_shape = model.input.shape[1:] # first shape element is batch size so don't copy it
    inputs = tf.keras.Input(shape=input_shape)

    # copy the layers - except the input layer - from original model
    layers = [l for l in model.layers[1:]]

    x = inputs
    for i, layer in enumerate(layers):
        print("in iteration i: ",i," and input shape is: ", x.shape, " for layer: ", layer.name)
        #x = layer(x)
        if type(layer) == Dense:
            input = layer.input
            output = layer.output
            weights = layer.weights

            # weights of Dense has shape: (features_in, features_out)
            _, features_out = weights[0].shape
            #TODO: copy all other attributes. Consider using get_config() and from_config()
            dense_shift_layer = DenseShift(weights[0].shape[-1]) 

            x = dense_shift_layer(x)

        elif type(layer) == Conv2D:
            input = layer.input
            output = layer.output
            weights = layer.weights

            # weights of Conv2D has shape: (filter_height, filter_width, channels_in, channels_out)
            print("weights: ", weights[0].shape)
            filter_height, filter_width, _, channels_out = weights[0].shape.as_list()
            #TODO: copy all other attributes. Consider using get_config() and from_config()
            conv2d_shift_layer = Conv2DShift(filters=channels_out, kernel_size = (filter_height, filter_width)) 

            x = conv2d_shift_layer(x)

        else:
            x = layer(x)
    outputs = x

    model_converted = Model(inputs=inputs, outputs=outputs)
    #TODO: Copy other attributes such as learning rate, optimizer, etc.
    return model_converted


from tensorflow.keras.applications.mobilenet import MobileNet
model = MobileNet(weights='imagenet')

model_converted = convert_to_shift(model)
model_converted.summary()