### This file is deprecated, see model.py for all commands.

import csv
import cv2
import numpy as np
from sys import getsizeof
import math

def shuffle_sync(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

def get_data(folder, cam_adj):
    lines = []

    print("Loading training folder:", folder)

    with open(folder +'/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    print("training samples:", len(lines))

    images_src = []
    measurements = []

    #skip the first header line.
    for line in lines[1:]:
        measurement = float(line[3])
        prob = np.random.random()
        camera = 0
        if prob < 0.7 and (-0.01 < measurement or measurement < 0.01):
            # drop 0 angle data with 70% prbability:
            camera = np.random.choice(['left', 'right'])
                #for camera in range(3):


            if camera == "left":
                #print("left")
                camera = 1
                measurement += cam_adj
            if camera == "right":
                #print("right")
                camera = 2
                measurement -= cam_adj
            #if camera == 0:
                #print("center")
        source_path = line[camera]
        filename = source_path.split('/')[-1]
        current_path = folder +'/IMG/' + filename
        measurements.append(measurement)
        images_src.append(current_path)
    nb_training_images = len(images_src)
    print("training images:", nb_training_images)
    return images_src, measurements

def data_generator(X, y, batch_size=128, crop=None, normalize=None):
    #f = h5py.File("/tmp/testData.hdf5", "w")
    #dset = f.create_dataset("X_train", (160, 320, 3), dtype='float32')

    nb_images = len(X)
    print("Samples to batch:", nb_images)
    # iterate over the total number of batches necessary.
    total_batches = math.ceil(nb_images / batch_size)
    print("necessary_batches:", total_batches)
    #derivatives = 2

    height = 160
    width = 320
    batch_features = np.zeros((batch_size*2, height, width, 3))
    batch_labels = np.zeros((batch_size*2, 1))

    while 1:
        # augmented_images, augmented_measurements = [], []
        # for image,measurement in zip(images, measurements):
        #     augmented_images.append(image)
        #     augmented_measurements.append(measurement)
        #     #augmented_images.append(cv2.flip(image,1))
        #     #augmented_measurements.append(measurement*-1.0)
        shuffle_sync(X, y)
        for batch_i in range(total_batches):
            start = batch_i*batch_size
            end = start+batch_size
            #print('end', end)
            batch_X = X[start:end]
            batch_y = y[start:end]
            nb_batch_items = len(batch_X)
            # print("Starting batch:", batch_i)
            # print(".. Attempting From: {} To: {}".format(start, end))
            # print(".. Found items:", len(batch_X), len(batch_y))


            # if crop:
            #     height -= sum(crop[0])
            #     width -= sum(crop[1])
            # For every item in the source return the item and another item that's flipped horizontally,
            # and a weight that's flipped.
            # print("Processing batch items..")
            for j, (filename, measurement) in enumerate(zip(batch_X, batch_y)):
                # print("item:", j, measurement)

                #orig_idx = start+j
                #print("orig_idx", orig_idx)
                #flip_idx = j + nb_batch_items
                img = cv2.imread(filename)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # if crop:
                #     start_y = crop[0][0]
                #     stop_y = height + start_y
                #     start_x = crop[1][0]
                #     stop_x = width + start_x
                #     #print(start_y, stop_y, start_x, stop_x)
                #     img = img[start_y:stop_y, start_x: stop_x]
               #if normalize:
                #    img = img / 255.
                batch_features[(j*2)] = img
                batch_labels[(j*2)] = measurement
                batch_features[(j*2)+1] = cv2.flip(img, 1)
                batch_labels[(j*2)+1] = measurement*-1.0
            #print("yield")
            yield ((batch_features, batch_labels))