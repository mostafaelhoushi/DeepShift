'''Train VGG on the CIFAR10 small iamges dataset.
'''
# Path hack.
import sys, os
sys.path.insert(0, os.path.abspath('..'))

import argparse

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

from shift_layer import *
from convolutional_shift import *
from convert_to_shift import *

def cifar10_mobilenet(depth = 16, loss='categorical_crossentropy', shift_depth=0, epochs=250):
    tf.enable_eager_execution()

    batch_size = 32
    num_classes = 10

    # Model name, depth and version
    model_type = 'VGG%d' % (depth)

    # Get the data.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Convert class vectors to binary class matrices.
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    # Preprocess the images.
    x_train = tf.keras.applications.vgg16.preprocess_input(x_train)
    x_test = tf.keras.applications.vgg16.preprocess_input(x_test)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print('y_train shape:', y_train.shape)

    # Get the model and compile it.
    img_input = tf.keras.layers.Input(shape=(32, 32, 3))
    if depth == 16:
        model = VGG16(input_tensor=img_input, weights=None, classes=num_classes)
    elif depth == 19:
        model = VGG19(input_tensor=img_input, weights=None, classes=num_classes)
    else:
        raise ValueError("VGG depth should be either 16 or 19")

    # Convert layers to shift
    if shift_depth > 0:
        model = convert_to_shift(model, num_to_replace=shift_depth)

    model.compile(loss='categorical_crossentropy',
                optimizer=tf.train.AdamOptimizer(),
                metrics=['accuracy'])

    model.summary()

    # Prepare model model saving directory.
    model_name = 'cifar10_%s_model_shift_%s' % (model_type,shift_depth)
    model_dir = os.path.join(os.path.join(os.getcwd(), 'saved_models'), model_name)
    model_checkpoint_name = 'model' + '.{epoch:03d}.h5'
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    model_checkpoint_path = os.path.join(model_dir, model_checkpoint_name)

    model_summary_name = 'model_summary.txt'
    model_summary_path = os.path.join(model_dir, model_summary_name)
    with open(model_summary_path, 'w') as fp: 
        model.summary(print_fn=lambda x: fp.write(x + '\n'))

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=model_checkpoint_path,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True)

    csv_logger = CSVLogger(os.path.join(model_dir,"model"+ "_train_log.csv"))

    callbacks = [checkpoint, csv_logger]

    print("Training model.")
    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            shuffle=True,
            verbose=1,
            callbacks=callbacks)
            
    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    # TODO: obtain best accuracy model and save it separately

    # save weights to .csv files to verify
    for layer in model.layers:
        if isinstance(layer, Conv2DShift) or isinstance(layer, DenseShift):
            for index, w in enumerate(layer.get_weights()):
                weights_csv_name = layer.name + "_" + str(index) + ".csv"
                if len(w.shape) > 2:
                    w = np.reshape(np.transpose(w, (1,0,2,3)), (w.shape[0],-1))
                np.savetxt(os.path.join(model_dir, weights_csv_name), w, fmt="%1.4f", delimiter=",")

if __name__== "__main__":
    parser = argparse.ArgumentParser(description='Train VGG16 or VGG19 on CIFAR10 data.')
    
    parser.add_argument('--depth', type=int, default=16, help='Model depth - either 16 or 19 (default: 16)')
    parser.add_argument('--loss', default="categorical_crossentropy", help='loss (default: ''categorical_crossentropy'')')
    parser.add_argument('--shift_depth', type=int, default=0, help='number of shift conv layers from the end (default: 0)')
    parser.add_argument('--epochs', type=int, default=250, help='number of epochs to train (default: 250)')

    args = parser.parse_args()
    
    cifar10_mobilenet(args.depth, args.loss, args.shift_depth, args.epochs)