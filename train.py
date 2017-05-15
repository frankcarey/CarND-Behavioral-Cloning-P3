from sklearn.model_selection import train_test_split
import math
import utils

# TUnable Parameters
batch_size = 2048
epochs = 10
learning_rate = 0.0001
l2_reg = 0.0001
cam_adj = 1.0
nb_derivatives = 2
#train_folder = './data/example_training'
train_folder = './data/train_2'




X, y = utils.get_data(train_folder, cam_adj)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.noise import GaussianNoise
from keras import regularizers, optimizers

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((50, 20), (0,0))))

model.add(Conv2D(24, (5, 5), strides=(2, 2),  padding='same', activation="relu", kernel_regularizer=regularizers.l2(l2_reg)))
model.add(Conv2D(36, (5, 5), strides=(2, 2), padding='same', activation="relu", kernel_regularizer=regularizers.l2(l2_reg)))
model.add(Conv2D(48, (5, 5), strides=(2, 2), padding='same', activation="elu", kernel_regularizer=regularizers.l2(l2_reg)))
#model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation="elu", kernel_regularizer=regularizers.l2(l2_reg)))

#model.add(MaxPooling2D())
model.add(Flatten())
#model.add(Dropout(0.25))
model.add(Dense(100, kernel_regularizer=regularizers.l2(l2_reg), activation='elu'))
#model.add(Dropout(0.25))
model.add(Dense(50, kernel_regularizer=regularizers.l2(l2_reg), activation='elu'))
model.add(Dense(1, kernel_regularizer=regularizers.l2(l2_reg), activation='elu'))

train_steps = math.ceil(len(X_train)*nb_derivatives/batch_size)
valid_steps = math.ceil(len(X_test)*nb_derivatives/batch_size)

print(train_steps)

adam = optimizers.Adam(lr=0.001)
model.compile(loss='mse', optimizer=adam)
model.fit_generator(
        utils.data_generator(X_train, y_train),
        steps_per_epoch=train_steps,
        validation_data=utils.data_generator(X_test, y_test),
        validation_steps=valid_steps,
        epochs=epochs
)

model.save('./data/model.h5')

print('[DONE]')
