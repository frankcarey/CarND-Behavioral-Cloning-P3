import csv
import cv2
import numpy as np
import h5py
from sys import getsizeof
from sklearn.model_selection import train_test_split
import math


def get_data():
    lines = []

    with open('./data/example_training/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    print("training samples:", len(lines))

    images_src = []
    measurements = []

    #skip the first header line.
    for line in lines[1:]:
        for camera in range(3):
            source_path = line[camera]
            filename = source_path.split('/')[-1]
            current_path = './data/example_training/IMG/' + filename
            images_src.append(current_path)
            measurement = float(line[3])
            if camera == 1:
                measurement += 0.2
            if camera == 2:
                measurement -= 0.2
            measurements.append(measurement)
    nb_training_images = len(images_src)
    print("training images:", nb_training_images, getsizeof(images_src))
    return images_src, measurements

def data_generator(X, y, batch_size=32):
    #f = h5py.File("/tmp/testData.hdf5", "w")
    #dset = f.create_dataset("X_train", (160, 320, 3), dtype='float32')


    while True:
        # augmented_images, augmented_measurements = [], []
        # for image,measurement in zip(images, measurements):
        #     augmented_images.append(image)
        #     augmented_measurements.append(measurement)
        #     #augmented_images.append(cv2.flip(image,1))
        #     #augmented_measurements.append(measurement*-1.0)
        nb_images = len(X)
        for i in range((nb_images // batch_size) +1):
            batch_features = np.zeros((batch_size, 160, 320, 3))
            batch_labels = np.zeros((batch_size,1))
            for j, filename in enumerate(X[i*batch_size:(i+1)*batch_size]):
                idx = i+j
                batch_features[j] = cv2.imread(filename)
                batch_labels[j] = y[idx]
            yield batch_features, batch_labels

batch_size = 1024
X, y = get_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Conv2D(6, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(6, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))


train_steps = math.ceil(len(X_train)/batch_size)
valid_steps = math.ceil(len(X_test)/batch_size)

print(train_steps)

model.compile(loss='mse', optimizer="adam")
model.fit_generator(
        data_generator(X_train, y_train),
        steps_per_epoch=train_steps,
        validation_data=data_generator(X_test, y_test),
        validation_steps=valid_steps,
        epochs=10
)

model.save('./data/model.h5')

print('[DONE]')