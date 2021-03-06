import tensorflow as tf
import numpy as np
import random
import csv
import cv2
import math

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.layers import Activation, Dense, Dropout, ELU, Flatten, Input, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.models import Sequential, Model, load_model, model_from_json
from keras.regularizers import l2
from keras.models import load_model

def get_training_data(log_file, steering_offset):
    """
    Reads a csv file and returns two lists separated into examples and labels.
    :param log_file: The path of the log file to be read.

    Note: This version from https://github.com/ncondo
    """
    image_names, steering_angles = [], []
    # Steering offset used for left and right images

    with open(log_file, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        for center_img, left_img, right_img, angle, _, _, _ in reader:
            angle = float(angle)
            image_names.append([center_img.strip(), left_img.strip(), right_img.strip()])
            steering_angles.append([angle, angle+steering_offset, angle-steering_offset])

    return image_names, steering_angles


def generate_batch(X_train, y_train, samples=64):
    """
    Return two numpy arrays containing images and their associated steering angles.
        X_train: A list of image names to be read in from data directory.
        y_train: A list of steering angles associated with each image.
        samples: The size of the numpy arrays to be return on each pass.

    This is a simpler version as it only outputs the number of samples you request
    and it randomly samples over the training data that's passed in.
    Note: This simpler version from https://github.com/ncondo
    """
    images = np.zeros((samples, 160, 320, 3), dtype=np.float32)
    angles = np.zeros((samples,), dtype=np.float32)
    while True:
        straight_count = 0
        for i in range(samples):
            # Select a random index to use for data sample
            sample_index = random.randrange(len(X_train))
            image_index = random.randrange(len(X_train[0]))
            angle = y_train[sample_index][image_index]
            # Limit angles of less than absolute value of .1 to no more than 1/2 of data
            # to reduce bias of car driving straight
            if abs(angle) < .1:
                straight_count += 1
            if straight_count > (batch_size * .5):
                while abs(y_train[sample_index][image_index]) < .1:
                    sample_index = random.randrange(len(X_train))
            # Read image in from directory, process, and convert to numpy array
            img_path = train_folder + '/' + str(X_train[sample_index][image_index])
            image = cv2.imread(img_path)
            print(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = process_image(image)
            image = np.array(image, dtype=np.float32)
            # Flip image and apply opposite angle 50% of the time
            if random.randrange(2) == 1:
                image = cv2.flip(image, 1)
                angle = -angle
            images[i] = image
            angles[i] = angle
        yield images, angles

def normalize(image):
    """
    Returns a normalized image with feature values from -1.0 to 1.0.
    :param image: Image represented as a numpy array.
    """
    return image / 127.5 - 1.


def random_brightness(image):
    """
    Returns an image with a random degree of brightness.
    :param image: Image represented as a numpy array.

    Note: From https://github.com/ncondo
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    brightness = .25 + np.random.uniform()
    image[:,:,2] = image[:,:,2] * brightness
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image


def process_image(image):
    """
    Returns an image after applying several preprocessing functions.
    :param image: Image represented as a numpy array.
    """
    image = random_brightness(image)
    # Cropping and
    return image


def get_model():
    """
    Returns a compiled keras model ready for training.

    This version based on https://github.com/ncondo, which is the NVIDIA
    self-driving model. I updated it for keras 2 and to have cropping happen in the model.

    Differences that may matter:
    - I was using the new Conv2D() layers vs Convolution2D
    - Adding small amounts of dropout at the beginning, then up to 50% at the last FC layers.
    - He doesn't add any MaxPooling layers (unless that's somehow included)

    Seems irrelevant differences might be:
    - init='he_normal'
    - subsample=(2, 2) (I was using strides)
    - border_mode='valid' ( I was setting the padding)
    -  W_regularizer=l2(0.001) (I was trying with kernel_regularizer=regularizers.l2(l2_reg))

    """
    model = Sequential([
        # Normalize image to -1.0 to 1.0
        Lambda(normalize, input_shape=(160, 320, 3)),
        # Crop the image as part of the model by removing 40px off the top and 20px from the bottom of the image.
        Cropping2D(cropping=((40, 20), (0,0))),
        # Convolutional layer 1 24@31x98 | 5x5 kernel | 2x2 stride | elu activation
        Convolution2D(24, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2), init='he_normal', W_regularizer=l2(0.001)),
        # Dropout with drop probability of .1 (keep probability of .9)
        Dropout(.1),
        # Convolutional layer 2 36@14x47 | 5x5 kernel | 2x2 stride | elu activation
        Convolution2D(36, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2), init='he_normal', W_regularizer=l2(0.001)),
        # Dropout with drop probability of .2 (keep probability of .8)
        Dropout(.2),
        # Convolutional layer 3 48@5x22  | 5x5 kernel | 2x2 stride | elu activation
        Convolution2D(48, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2), init='he_normal', W_regularizer=l2(0.001)),
        # Dropout with drop probability of .2 (keep probability of .8)
        Dropout(.2),
        # Convolutional layer 4 64@3x20  | 3x3 kernel | 1x1 stride | elu activation
        Convolution2D(64, 3, 3, border_mode='valid', activation='elu', subsample=(1, 1), init='he_normal', W_regularizer=l2(0.001)),
        # Dropout with drop probability of .2 (keep probability of .8)
        Dropout(.2),
        # Convolutional layer 5 64@1x18  | 3x3 kernel | 1x1 stride | elu activation
        Convolution2D(64, 3, 3, border_mode='valid', activation='elu', subsample=(1, 1), init='he_normal', W_regularizer=l2(0.001)),
        # Flatten
        Flatten(),
        # Dropout with drop probability of .3 (keep probability of .7)
        Dropout(.3),
        # Fully-connected layer 1 | 100 neurons | elu activation
        Dense(100, activation='elu', init='he_normal', W_regularizer=l2(0.001)),
        # Dropout with drop probability of .5
        Dropout(.5),
        # Fully-connected layer 2 | 50 neurons | elu activation
        Dense(50, activation='elu', init='he_normal', W_regularizer=l2(0.001)),
        # Dropout with drop probability of .5
        Dropout(.5),
        # Fully-connected layer 3 | 10 neurons | elu activation
        Dense(10, activation='elu', init='he_normal', W_regularizer=l2(0.001)),
        # Dropout with drop probability of .5
        Dropout(.5),
        # Output
        Dense(1, activation='linear', init='he_normal')
    ])

    model.compile(optimizer='adam', loss='mse')
    return model

if __name__=="__main__":

    # Tunable Parameters
    batch_size = 64 # It turns out that setting this to a higher number (512 for instance) makes the model converge SLOWER!
    epochs = 28 #20
    #train_folder = './data/example_training'
    train_folder = './data/train_3'
    model_path = 'model.h5'
    steering_offset = 0.275
    random_state = 14
    retrain = True
    test_split = 0.1

    # Get the training data from log file, shuffle, and split into train/validation datasets
    X_train, y_train = get_training_data(train_folder + "/driving_log.csv", steering_offset)
    X_train, y_train = shuffle(X_train, y_train, random_state=random_state)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=test_split, random_state=random_state)

    train_steps = math.ceil(len(X_train)/batch_size)
    valid_steps = math.ceil(len(X_validation)/batch_size)

    print("Training steps per epoch: ")

    # Get model either a new one (compiled) or load an existing one for further training.
    if retrain == True:
        model = load_model(model_path)
    else:
        model = get_model()

    # Print the summmary of layers and params.
    model.summary()

    # Fit using a generator.
    model.fit_generator(generate_batch(X_train, y_train),
                        steps_per_epoch=train_steps,
                        validation_data=generate_batch(X_validation, y_validation, samples=batch_size),
                        validation_steps=valid_steps,
                        epochs=epochs
                        )#, callbacks=[early_stop])

    print('Saving model..')
    model.save(model_path)

    # Explicitly end tensorflow session
    # Not sure if this is still needed?
    from keras import backend as K

    K.clear_session()