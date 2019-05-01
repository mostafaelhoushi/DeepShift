import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import cv2 as cv
from keras.preprocessing import image
from keras.applications import mobilenet, resnet50
from keras.applications import MobileNet, MobileNetV2
from keras.utils import to_categorical

from enum import Enum
import distutils

from cifar10_resshift import *
from convert_to_shift import *

class ModelPreprocessor(Enum):
    RESNET = 1
    MOBILENET = 2
    OTHER = -1


def resize_images(images, new_size):
    shape = images.shape
    new_shape = ((shape[0],) + new_size + (shape[-1],))
    new_images = np.empty(shape=new_shape) 
    for idx in range(shape[0]):
        new_images[idx] = cv.resize(images[idx], new_size)
        
    return new_images
    
def preprocess_fn(image, label, model_preprocess, target_shape=(224,224), num_classes=1000):
    '''A transformation function to preprocess raw data
    into trainable input. '''
    x = resize_images(np.expand_dims(image,0), target_shape)
    if (model_preprocess is ModelPreprocessor.RESNET):
        x = resnet50.preprocess_input(x)
    elif (model_preprocess is ModelPreprocessor.MOBILENET):
        x = mobilenet.preprocess_input(x)
    elif not None:
        x = x/255
    
    y = to_categorical(label, num_classes=num_classes)
    y = np.expand_dims(y, 0)
    return x, y

def imagenet_generator(dataset, batch_size=32, num_classes=1000, is_training=False):
    images = np.zeros((batch_size, 224, 224, 3))
    labels = np.zeros((batch_size, num_classes))
    while True:
        count = 0 
        for sample in tfds.as_numpy(dataset):
            images[count%batch_size], labels[count%batch_size] = preprocess_fn(sample["image"], sample["label"], ModelPreprocessor.MOBILENET, num_classes=1000)
            count += 1
            if (count%batch_size == 0):
                yield images, labels

def imagenet_mobilenet(version = 1, loss='categorical_crossentropy', shift_depth=0, epochs=20, desc="", freeze=True):
    tf.enable_eager_execution()

    # Training parameters
    batch_size = 32  # orig paper trained all networks with batch_size=128
    num_classes = 1000

    # Load the Imagenet 2012 data.
    # Fetch the dataset directly
    imagenet = tfds.image.Imagenet2012()
    ## or by string name
    #imagenet = tfds.builder('imagenet2012')

    # Describe the dataset with DatasetInfo
    C = imagenet.info.features['label'].num_classes
    Ntrain = imagenet.info.splits['train'].num_examples
    Nvalidation = imagenet.info.splits['validation'].num_examples
    Nbatch = 32
    assert C == 1000
    assert Ntrain == 1281167
    assert Nvalidation == 50000

    # Download the data, prepare it, and write it to disk
    imagenet.download_and_prepare()

    # Load data from disk as tf.data.Datasets
    datasets = imagenet.as_dataset()
    train_dataset, validation_dataset = datasets['train'], datasets['validation']
    assert isinstance(train_dataset, tf.data.Dataset)
    assert isinstance(validation_dataset, tf.data.Dataset)

    # Input image dimensions.
    input_shape = (224, 224, 3)

    # Get the model and compile it.
    if version == 1:
        model = MobileNet(weights='imagenet')
        model_type = "MobileNet"
    elif version == 2:
        model = MobileNetV2(weights='imagenet')
        model_type = "MobileNetV2"
    else:
        raise ValueError("version should be either 1 or 2")

    # Convert layers to shift
    if shift_depth > 0:
        model = convert_to_shift(model, num_to_replace=shift_depth, convert_weights=True, freeze=freeze)

    model.compile(loss='categorical_crossentropy',
                optimizer=tf.train.AdamOptimizer(),
                metrics=['accuracy'])

    model.summary()

    # Prepare model model saving directory.
    if desc is not None and len(desc) > 0:
        model_name = 'imagenet/%s/%s_shift_%s' % (model_type,desc,shift_depth)
    else:
        model_name = 'imagenet/%s/shift_%s' % (model_type,shift_depth)
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

    if epochs > 0:
        print("Training model.")
        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(imagenet_generator(train_dataset, batch_size=Nbatch, is_training=True),
                            validation_data=imagenet_generator(validation_dataset, batch_size=Nbatch),
                            validation_steps = Nvalidation // Nbatch,
                            epochs=epochs, verbose=1, steps_per_epoch=Ntrain // Nbatch,
                            callbacks=callbacks)

    # TODO: obtain best accuracy model and save it separately

    # Infer model
    print("Inferring model.")
    scores = model.evaluate_generator(imagenet_generator(validation_dataset,batch_size=32), 
                                    steps= Nvalidation // Nbatch, 
                                    verbose=1)
    print('Inference loss:', scores[0])
    print('Inference accuracy:', scores[1])

    # save weights to .csv files to verify
    for layer in model.layers:
        if isinstance(layer, Conv2DShift) or isinstance(layer, DenseShift):
            for index, w in enumerate(layer.get_weights()):
                weights_csv_name = layer.name + "_" + str(index) + ".csv"
                if len(w.shape) > 2:
                    w = np.reshape(np.transpose(w, (1,0,2,3)), (w.shape[0],-1))
                np.savetxt(os.path.join(model_dir, weights_csv_name), w, fmt="%1.4f", delimiter=",")

if __name__== "__main__":
    parser = argparse.ArgumentParser(description='Train MobileNet or MobileNetv2 on Imagenet2012 data.')
    
    # TODO: make default values constant variables instead of hard-coding
    parser.add_argument('--version', type=int, default=1, help='Model version (default: 1)')
    parser.add_argument('--loss', default="categorical_crossentropy", help='loss (default: ''categorical_crossentropy'')')
    parser.add_argument('--shift_depth', type=int, default=0, help='number of shift conv layers from the end (default: 0)')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 20)')
    parser.add_argument('--desc', default="", help="description to append to model directory")
    parser.add_argument('--freeze', default=True, type=lambda x:bool(distutils.util.strtobool(x)), help='freeze weights of all layers upto the first converted layer (default: True)')

    args = parser.parse_args()
    
    imagenet_mobilenet(args.version, args.loss, args.shift_depth, args.epochs, args.desc, args.freeze)
