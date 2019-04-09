'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import tensorflow as tf

from shift_layer import *
from round_fixed import *
import sys
######necessary packages to use spfpm, the fixed point package
sys.path.insert(0, '../spfpm/')
from FixedPoint import FXfamily, FXnum
######
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

tf.enable_eager_execution()

batch_size = 128
num_classes = 10
epochs = 20

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(DenseShift(512, input_shape=(784,)))
#model.add(RoundToFixed())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512, name='dense_shift_2'))
model.add(Activation('relu', name='relu2'))
model.add(Dropout(0.2))
#model.add(RoundToFixed(name='round2fix_2'))
model.add(DenseShift(num_classes, name='dense_shift_3'))
model.add(Activation('softmax', name='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=tf.train.RMSPropOptimizer(learning_rate=0.01),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
#sample code on using fixed point package
x = FXnum(22)
print("========",x,"===========")

for layer in model.layers:
    print("Layer: " + layer.name)
    for index, w in enumerate(layer.get_weights()):
        np.savetxt(layer.name + "_" + str(index) + ".csv", w, delimiter=",")