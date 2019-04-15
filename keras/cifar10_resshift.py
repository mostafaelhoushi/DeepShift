"""
#Trains a ResNet on the CIFAR10 dataset.
ResNet v1:
[Deep Residual Learning for Image Recognition
](https://arxiv.org/pdf/1512.03385.pdf)
ResNet v2:
[Identity Mappings in Deep Residual Networks
](https://arxiv.org/pdf/1603.05027.pdf)
Model|n|200-epoch accuracy|Original paper accuracy |sec/epoch GTX1080Ti
:------------|--:|-------:|-----------------------:|---:
ResNet20   v1|  3| 92.16 %|                 91.25 %|35
ResNet32   v1|  5| 92.46 %|                 92.49 %|50
ResNet44   v1|  7| 92.50 %|                 92.83 %|70
ResNet56   v1|  9| 92.71 %|                 93.03 %|90
ResNet110  v1| 18| 92.65 %|            93.39+-.16 %|165
ResNet164  v1| 27|     - %|                 94.07 %|  -
ResNet1001 v1|N/A|     - %|                 92.39 %|  -
&nbsp;
Model|n|200-epoch accuracy|Original paper accuracy |sec/epoch GTX1080Ti
:------------|--:|-------:|-----------------------:|---:
ResNet20   v2|  2|     - %|                     - %|---
ResNet32   v2|N/A| NA    %|            NA         %| NA
ResNet44   v2|N/A| NA    %|            NA         %| NA
ResNet56   v2|  6| 93.01 %|            NA         %|100
ResNet110  v2| 12| 93.15 %|            93.63      %|180
ResNet164  v2| 18|     - %|            94.54      %|  -
ResNet1001 v2|111|     - %|            95.08+-.14 %|  -
"""

from __future__ import print_function
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten, DepthwiseConv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
import numpy as np
import os
import argparse

from shift_layer import *
from convolutional_shift import *

tf.enable_eager_execution()

# Script Arguments: n, version
# n: Model parameter
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------

# version: Model version
# Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)

def cifar10_resnet(n = 3, version = 1, loss='categorical_crossentropy', shift_depth=0):
	# Training parameters
	batch_size = 32  # orig paper trained all networks with batch_size=128
	epochs = 200
	data_augmentation = True
	num_classes = 10

	# Subtracting pixel mean improves accuracy
	subtract_pixel_mean = True

	# Computed depth from supplied model parameter n
	if version == 1:
		depth = n * 6 + 2
	elif version == 2:
		depth = n * 9 + 2

	# Model name, depth and version
	model_type = 'ResNet%dv%d' % (depth, version)

	# Load the CIFAR10 data.
	(x_train, y_train), (x_test, y_test) = cifar10.load_data()

	# Input image dimensions.
	input_shape = x_train.shape[1:]

	# Normalize data.
	x_train = x_train.astype('float32') / 255
	x_test = x_test.astype('float32') / 255

	# If subtract pixel mean is enabled
	if subtract_pixel_mean:
		x_train_mean = np.mean(x_train, axis=0)
		x_train -= x_train_mean
		x_test -= x_train_mean

	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')
	print('y_train shape:', y_train.shape)

	# Convert class vectors to binary class matrices.
	y_train = tf.keras.utils.to_categorical(y_train, num_classes)
	y_test = tf.keras.utils.to_categorical(y_test, num_classes)

	if version == 2:
		model = resnet_v2(input_shape=input_shape, depth=depth)
	else:
		model = resnet_v1(input_shape=input_shape, depth=depth, shift_depth=shift_depth)

	model.compile(loss=loss,
				  optimizer=tf.train.AdamOptimizer(learning_rate=lr_schedule(0)),
				  metrics=['accuracy'])
	model.summary()
	print(model_type)

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

	# TODO: fix lr_reducer and lr_scheduler for eager mode
	lr_scheduler = LearningRateScheduler(lr_schedule)

	'''
	# Not compatible with eager execution
	lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
								   cooldown=0,
								   patience=5,
								   min_lr=0.5e-6)
	'''
								   
	csv_logger = CSVLogger(os.path.join(model_dir,"model"+ "_train_log.csv"))

<<<<<<< HEAD
	if shift_depth==0:
		callbacks = [checkpoint, lr_reducer, lr_scheduler, csv_logger]
	else:
		callbacks = [checkpoint, lr_reducer, csv_logger]
=======
	# callbacks = [checkpoint, lr_reducer, lr_scheduler, csv_logger]
	callbacks = [checkpoint, csv_logger]
>>>>>>> f67f29b... get conv and resnet to work in eager execution mode

	# Run training, with or without data augmentation.
	if not data_augmentation:
		print('Not using data augmentation.')
		model.fit(x_train, y_train,
				  batch_size=batch_size,
				  epochs=epochs,
				  validation_data=(x_test, y_test),
				  shuffle=True,
				  callbacks=callbacks)
	else:
		print('Using real-time data augmentation.')
		# This will do preprocessing and realtime data augmentation:
		datagen = ImageDataGenerator(
			# set input mean to 0 over the dataset
			featurewise_center=False,
			# set each sample mean to 0
			samplewise_center=False,
			# divide inputs by std of dataset
			featurewise_std_normalization=False,
			# divide each input by its std
			samplewise_std_normalization=False,
			# apply ZCA whitening
			zca_whitening=False,
			# epsilon for ZCA whitening
			zca_epsilon=1e-06,
			# randomly rotate images in the range (deg 0 to 180)
			rotation_range=0,
			# randomly shift images horizontally
			width_shift_range=0.1,
			# randomly shift images vertically
			height_shift_range=0.1,
			# set range for random shear
			shear_range=0.,
			# set range for random zoom
			zoom_range=0.,
			# set range for random channel shifts
			channel_shift_range=0.,
			# set mode for filling points outside the input boundaries
			fill_mode='nearest',
			# value used for fill_mode = "constant"
			cval=0.,
			# randomly flip images
			horizontal_flip=True,
			# randomly flip images
			vertical_flip=False,
			# set rescaling factor (applied before any other transformation)
			rescale=None,
			# set function that will be applied on each input
			preprocessing_function=None,
			# image data format, either "channels_first" or "channels_last"
			data_format=None,
			# fraction of images reserved for validation (strictly between 0 and 1)
			validation_split=0.0)

		# Compute quantities required for featurewise normalization
		# (std, mean, and principal components if ZCA whitening is applied).
		datagen.fit(x_train)

		# Fit the model on the batches generated by datagen.flow().
		model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
							validation_data=(x_test, y_test),
							epochs=epochs, verbose=1, workers=4, steps_per_epoch=x_train.shape[0]//batch_size,
							callbacks=callbacks, use_multiprocessing=True)

	# Score trained model.
	scores = model.evaluate(x_test, y_test, verbose=1)
	print('Test loss:', scores[0])
	print('Test accuracy:', scores[1])

    # TODO: obtain best accuracy model and save it separately

    # save weights to .csv files to verify
	for layer in model.layers:
		print("Layer: " + layer.name)
		if isinstance(layer, Conv2DShift) or isinstance(layer, DenseShift):
			for index, w in enumerate(layer.get_weights()):
				weights_csv_name = layer.name + "_" + str(index) + ".csv"
				if len(w.shape) > 2:
					w = np.reshape(np.transpose(w, (1,0,2,3)), (w.shape[0],-1))
				np.savetxt(os.path.join(model_dir, weights_csv_name), w, fmt="%1.4f", delimiter=",")


def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True,
                 use_shift=False):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """

    if use_shift == False:
        conv = Conv2D(num_filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding='same',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-4))
    else:
        conv = Conv2DShift(num_filters,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding='same',
                        kernel_regularizer=None) #l2_powerof2(1e-4)) #TODO: fix regularizer in eager mode

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=10, shift_depth=0):
    """ResNet Version 1 Model builder [a]
    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)

    conv_so_far = 0
    use_shift = False

    conv_so_far += 1
    use_shift = (depth - conv_so_far <= shift_depth)
    x = resnet_layer(inputs=inputs, use_shift=use_shift)

    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            
            conv_so_far += 1
            use_shift = (depth - conv_so_far <= shift_depth)
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides,
                             use_shift=use_shift)

            conv_so_far += 1
            use_shift = (depth - conv_so_far <= shift_depth)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None,
                             use_shift=use_shift)

            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                conv_so_far += 1
                use_shift = (depth - conv_so_far <= shift_depth)
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False,
                                 use_shift=use_shift)

            x = tf.keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    #outputs = Dense(num_classes, 
    #                activation='softmax',
    #                kernel_initializer='he_normal')(y)
    y = DenseShift(num_classes)(y)
    outputs = Activation('softmax')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_v2(input_shape, depth, num_classes=10):
    """ResNet Version 2 Model builder [b]
    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    #outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(y)
    y = DenseShift(num_classes)(y)
    outputs = Activation('softmax')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

if __name__== "__main__":
	parser = argparse.ArgumentParser(description='Train various versions of ResNet on CIFAR10 data.')
	
	parser.add_argument('--n', type=int, default=3, help='Model parameter (default: 3)')
	parser.add_argument('--version', type=int, default=1, help='Model version (default: 1)')
	parser.add_argument('--loss', default="categorical_crossentropy", help='loss (default: ''categorical_crossentropy'')')
	parser.add_argument('--shift_depth', type=int, default=0, help='number of shift conv layers from the end (default: 0)')

	args = parser.parse_args()
	
	cifar10_resnet(args.n, args.version, args.loss, args.shift_depth)