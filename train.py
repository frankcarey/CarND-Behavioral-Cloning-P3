### This file is deprecated, see model.py for all commands.

from sklearn.model_selection import train_test_split
import math
import utils

# TUnable Parameters
batch_size = 64 #2048
epochs = 3 #20
learning_rate = None #0.0005
l2_reg = 0.0001
cam_adj = 0.10
# nb_derivatives = 2
train_folder = './data/example_training'
#train_folder = './data/train_3'
act="elu"
pad="valid"

X, y = utils.get_data(train_folder, cam_adj)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print("Source training samples:", len(X_train), len(y_train) )
print("Source validation samples:", len(X_test), len(y_test))

train_steps = math.ceil(len(X_train)/batch_size)
valid_steps = math.ceil(len(X_test)/batch_size)
print("train_steps", train_steps)
print("valid_steps", valid_steps)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.noise import GaussianNoise
from keras import regularizers, optimizers

model = Sequential()
model.add(Lambda(lambda x: x / 127.5 -1, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((50, 20), (0,0))))
#model.add(GaussianNoise(0.1))
model.add(Conv2D(24, 5, strides=1, padding=pad, activation=act))
#model.add(MaxPooling2D())
model.add(Conv2D(36, 5, strides=1, padding=pad, activation=act))
#model.add(MaxPooling2D())
#model.add(Dropout(0.25))
# Works somewhat but twitchy with one layer, trying two
# model.add(Conv2D(5, 2, strides=1, padding=pad, activation=act))
# model.add(MaxPooling2D())
# model.add(Dropout(0.25))
# model.add(Conv2D(32, 5, strides=1, padding=pad, activation=act))
# model.add(MaxPooling2D())
# model.add(Dropout(0.25))
# model.add(Conv2D(48, 5, strides=(2, 2), padding=pad, activation=act, kernel_regularizer=regularizers.l2(l2_reg)))
# model.add(Dropout(0.25))
# model.add(Conv2D(64, 3, strides=(1, 1), padding=pad, activation=act, kernel_regularizer=regularizers.l2(l2_reg)))
# model.add(Dropout(0.25))
# model.add(Conv2D(64, 3, strides=(1, 1), padding=pad, activation=act, kernel_regularizer=regularizers.l2(l2_reg)))
# model.add(Dropout(0.25))

#model.add(MaxPooling2D())
model.add(Flatten())
# model.add(Dropout(0.25))
model.add(Dense(100, activation=act, kernel_regularizer=regularizers.l2(l2_reg)))
# model.add(Dropout(0.25))
model.add(Dense(50, activation=act))
# model.add(Dropout(0.25))
model.add(Dense(10, activation=act))
#model.add(Dropout(0.25))
#model.add(GaussianNoise(0.1))
model.add(Dense(1, activation=act))



# adam = optimizers.Adam(
#     #lr=learning_rate
# )
model.compile(loss='mse', optimizer='adam')
model.fit_generator(
        utils.data_generator(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=train_steps,
        validation_data=utils.data_generator(X_test, y_test, batch_size=batch_size),
        validation_steps=valid_steps,
        epochs=epochs
)

model.save('./data/model.h5')

print('[DONE]')
