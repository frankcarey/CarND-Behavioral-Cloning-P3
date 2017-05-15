import csv
import cv2
import numpy as np
from sys import getsizeof
import math

def get_data(folder='./data/train_1', cam_adj=0.3):
    lines = []

    with open(folder +'/driving_log.csv') as csvfile:
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
            current_path = folder +'/IMG/' + filename
            measurement = float(line[3])
            if camera == 1:
               measurement += cam_adj
            if camera == 2:
               measurement -= cam_adj
            # if camera != 0:
            #     break
            print(camera, measurement)
            measurements.append(measurement)
            images_src.append(current_path)
    nb_training_images = len(images_src)
    print("training images:", nb_training_images, getsizeof(images_src))
    return images_src, measurements

def data_generator(X, y, batch_size=32, crop=None, normalize=None):
    #f = h5py.File("/tmp/testData.hdf5", "w")
    #dset = f.create_dataset("X_train", (160, 320, 3), dtype='float32')


    while 1:
        # augmented_images, augmented_measurements = [], []
        # for image,measurement in zip(images, measurements):
        #     augmented_images.append(image)
        #     augmented_measurements.append(measurement)
        #     #augmented_images.append(cv2.flip(image,1))
        #     #augmented_measurements.append(measurement*-1.0)
        nb_images = len(X)
        # iterate over the total number of batches necessary.
        total_batches = math.ceil(nb_images / batch_size)
        derivatives = 1
        for batch_i in range(total_batches):
            start = batch_i*batch_size
            #print("Start", start)
            end = start+batch_size
            #print('end', end)
            batch_items = X[start:end]
            nb_batch_items = len(batch_items)

            height = 160
            width = 320
            if crop:
                height -= sum(crop[0])
                width -= sum(crop[1])
            batch_features = np.zeros((nb_batch_items*derivatives, height, width, 3))
            batch_labels = np.zeros((nb_batch_items*derivatives,1))

            # For every item in the source return the item and another item that's flipped horizontally,
            # and a weight that's flipped.
            for j, filename in enumerate(batch_items):
                orig_idx = start+j
                #print("orig_idx", orig_idx)
                #flip_idx = j + nb_batch_items
                img = cv2.imread(filename)
                if crop:
                    start_y = crop[0][0]
                    stop_y = height + start_y
                    start_x = crop[1][0]
                    stop_x = width + start_x
                    #print(start_y, stop_y, start_x, stop_x)
                    img = img[start_y:stop_y, start_x: stop_x]
                if normalize:
                    img = img / 255. 
                batch_features[j] = img
                batch_labels[j] = y[orig_idx]
                #batch_features[flip_idx] = cv2.flip(batch_features[j], 1)
                #batch_labels[flip_idx] = batch_labels[j]*-1.0
            #print("yield")
            yield (batch_features, batch_labels)