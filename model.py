# %% IMPORTS
from keras.utils import plot_model
import matplotlib.pyplot as plt
import multi_gpu as mgpu
from keras.callbacks import EarlyStopping
import csv
import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, BatchNormalization
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras import optimizers
from layers import Grayscale
import h5py

# %% INPUT LOADING FUNCTIONS
input_dir = './input_v1/'
output_dir = './output/'
if os.name == 'nt':
    input_dir = '.\\input_v1\\'
    output_dir = '.\\output\\'


class Data:
    X_train = None
    y_train = None

    def __init__(self, images, measurments):
        self.X_train = images
        self.y_train = measurments

    @classmethod
    def fromLists(self, images, measurments):
        X_train = np.array(images)
        y_train = np.array(measurments)
        return self(X_train, y_train)

    @classmethod
    def fromFile(self):
        with h5py.File(input_dir + 'input.h5', 'r') as hf:
            X_train = hf['X_train'][:]
            y_train = hf['y_train'][:]
            return self(X_train, y_train)

    def save(self):
        with h5py.File(input_dir + 'input.h5', 'w') as hf:
            hf.create_dataset("X_train",  data=self.X_train)
            hf.create_dataset("y_train",  data=self.y_train)

class Input:
    images = []
    measurments = []

    def add_sample(self, image_src, measurment):
        image = cv2.imread(image_src)
        self.__add(image, measurment)
        #Add flipped image
        # self.__add(np.fliplr(image), -measurment)

    def __add(self, image, measurment):
        self.images.append(image)
        self.measurments.append(measurment)

    def data(self):
        return Data.fromLists(self.images, self.measurments)


def read_input(src=input_dir + 'driving_log.csv'):
    input = Input()
    print('reading csv from ' + src)
    with open(src) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            steering_center = float(line[3])
            # create adjusted steering measurements for the side camera images
            correction = 0.25  # parameter to tune
            steering_left = steering_center + correction
            steering_right = steering_center - correction

            image_center = line[0]
            input.add_sample(image_center, steering_center)

            image_left = line[1]
            input.add_sample(image_left, steering_left)

            image_right = line[2]
            input.add_sample(image_right, steering_right)

    return input


def plot_loss(history_object, to_file):
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig(to_file)


def save_model(model, to_file):
    with open(to_file + '.yaml', "w") as yaml_file:
        yaml_file.write(model.to_yaml())
    model.save(to_file + '.h5')


# %% LOAD DATA FROM CSV
# input = read_input()
# data = input.data()
# data.save()

# %% LOAD DATA FROM NPY
data = Data.fromFile()

# %% MAKE MODEL
model = Sequential()
model.add(Cropping2D(cropping=((32, 20), (0, 0)), input_shape=(160, 320, 3)))
model.add(Grayscale())
model.add(
    BatchNormalization(axis=1)
)  #same as: model.add(Lambda(lambda x: x / 127 - 1, input_shape=(160, 320, 1)))
model.add(Convolution2D(24, (5, 5), strides=(2, 2), activation='relu'))
model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(Convolution2D(48, (5, 5), strides=(2, 2), activation='relu'))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
plot_model(model, to_file=output_dir + 'model.png', show_shapes=True)

# %% COMPILE MODEL
# model = mgpu.make_parallel(model,2) #enable parallel gpu
adam = optimizers.adam(lr=0.0001)
model.compile(loss='mse', optimizer=adam)

# %% RUN MODEL
earlyStopping = EarlyStopping(
    monitor='val_loss', min_delta=0.001, patience=2, verbose=0, mode='auto')

history_object = model.fit(
    data.X_train,
    data.y_train,
    validation_split=0.2,
    shuffle=True,
    epochs=10,
    callbacks=[earlyStopping])

# %% SAVE MODEL
plot_loss(history_object, to_file=output_dir + 'loss.png')
save_model(model, to_file=output_dir + 'model')

# TO START SERVER DO CALL
# python drive.py output\model
