'''Train MobileNet on the CIFAR10 small iamges dataset.
Inspired by:
    https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py
'''
# Path hack.
import sys, os
sys.path.insert(0, os.path.abspath('..'))

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications.mobilenet import MobileNet

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
model = MobileNet(input_tensor=img_input, weights=None, classes=num_classes)
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
