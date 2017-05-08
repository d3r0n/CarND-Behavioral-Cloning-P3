# %% IMPORTS & SETTINGS
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.layers import (ELU, BatchNormalization, Convolution2D, Cropping2D,
                          Dense, Dropout, Flatten, Lambda, MaxPooling2D)
from keras.models import Model, Sequential
from keras.utils import plot_model

import multi_gpu as mgpu
from data import BatchGenerator, Data, Input


def plot_loss(history_object, to_file):
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig(to_file)


def save_model(model, to_file):
    model.save(to_file + '.h5')
    with open(to_file + '.yaml', "w") as yaml_file:
        yaml_file.write(model.to_yaml())


# %% LOAD AND SAVE INPUT DATA
input_dir = './input_v1/'
output_dir = './output/'
input = Input.from_file(input_dir)
data = Data(input)
# data.save(input_dir)

# %% LOAD INPUT DATA
# input_dir = './input_v1/'
# output_dir = './output/'
# data = Data.from_file(input_dir)

# %% MAKE MODEL
model = Sequential()
model.add(Cropping2D(cropping=((32, 20), (0, 0)), input_shape=(160, 320, 3)))
#model.add(Grayscale())
model.add(
    Convolution2D(1, (1, 1), name='color_space_convolution', activation='elu'))
#BatchNormalization gives same result as using:
#model.add(Lambda(lambda x: x / 127 - 1, input_shape=(160, 320, 1)))
model.add(BatchNormalization(axis=1, name='pixel_normalization'))
model.add(Convolution2D(24, (5, 5), strides=(2, 2), activation='elu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.9))
model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation='elu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.8))
model.add(Convolution2D(48, (5, 5), strides=(2, 2), activation='elu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.7))
model.add(Convolution2D(64, (3, 3), activation='elu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Convolution2D(64, (3, 3), activation='elu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(1164))
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))
plot_model(model, to_file=output_dir + 'model.png', show_shapes=True)

# %% COMPILE MODEL
# model = mgpu.make_parallel(model,2) #enable parallel gpu
adam = optimizers.adam(lr=0.0001)
model.compile(loss='mse', optimizer=adam)

# %% RUN MODEL
earlyStopping = EarlyStopping(
    monitor='val_loss', min_delta=0.001, patience=5, verbose=1, mode='auto')
train_generator = BatchGenerator(data, 'training')
valid_generator = BatchGenerator(data, 'validation')

history_object = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.steps,
    validation_data=valid_generator,
    validation_steps=valid_generator.steps,
    epochs=1,
    callbacks=[earlyStopping],
    max_q_size=100,
    workers=10,
    pickle_safe=False)

# %% SAVE MODEL
plot_loss(history_object, to_file=output_dir + 'loss.png')
save_model(model, to_file=output_dir + 'model')

# TO START SERVER DO CALL
# python drive.py output\model
