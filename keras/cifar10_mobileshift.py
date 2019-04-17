'''Train MobileNet on the CIFAR10 small iamges dataset.
Inspired by:
    https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py
'''
# Path hack.
import sys, os
sys.path.insert(0, os.path.abspath('..'))

import argparse

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import MobileNet, MobileNetV2

def cifar10_mobilenet(version = 1, loss='categorical_crossentropy', shift_depth=0):
    tf.enable_eager_execution()

    batch_size = 32
    num_classes = 10
    epochs = 500

    # Get the data.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Convert class vectors to binary class matrices.
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    # Preprocess the images.
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # Get the model and compile it.
    img_input = tf.keras.layers.Input(shape=(32, 32, 3))
    if version == 1:
        model = MobileNet(input_tensor=img_input, weights=None, classes=num_classes)
    elif version == 2:
        model = MobileNetV2(input_tensor=img_input, weights=None, classes=num_classes)
    else:
        raise ValueError("version should be either 1 or 2")

    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    print("Training model.")
    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            shuffle=True,
            verbose=1)
            
    model.save('mobilenet_cifar10.h5')

if __name__== "__main__":
	parser = argparse.ArgumentParser(description='Train MobileNet or MobileNetv2 on CIFAR10 data.')
	
	parser.add_argument('--version', type=int, default=1, help='Model version (default: 1)')
	parser.add_argument('--loss', default="categorical_crossentropy", help='loss (default: ''categorical_crossentropy'')')
	parser.add_argument('--shift_depth', type=int, default=0, help='number of shift conv layers from the end (default: 0)')

	args = parser.parse_args()
	
	cifar10_mobilenet(args.version, args.loss, args.shift_depth)