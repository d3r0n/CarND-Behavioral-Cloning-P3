# %% IMPORTS & SETTINGS
import os
os.environ["CUDA_DEVICE_ORDER"] = "blahblah" # it will not find device and rollback to CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import matplotlib
matplotlib.use('Agg') #without this you need X server to run. FYI Amazon will fail.
import matplotlib.pyplot as plt #this must be after matplotlib.use('Agg') !
import numpy as np
import tensorflow as tf
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
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

# %% LOAD AND SAVE INPUT DATA
input_dir = './input_v7/'
output_dir = './output_v7/'
load_model = 'model.176-0.0290.h5'
input = Input.from_file(input_dir)
data = Data(input)
# data.save(input_dir)
# data = Data.from_file(input_dir)

# %% MAKE MODEL
def make_model():
    model = Sequential()
    #BatchNormalization gives same result as using:
    #model.add(Lambda(lambda x: x / 127 - 1, input_shape=(160, 320, 3)))
    model.add(BatchNormalization(axis=1, name='pixel_normalization', input_shape=(106, 320, 3)))
    #Conv 1
    model.add(Convolution2D(24, (5, 5), strides=(2, 2), padding='valid', activation='relu'))
    model.add(Dropout(0.2))
    #Conv 2
    model.add(Convolution2D(36, (5, 5), strides=(2, 2), padding='valid', activation='relu'))
    model.add(Dropout(0.3))
    #Conv 3
    model.add(Convolution2D(48, (5, 5), strides=(2, 2), padding='valid', activation='relu'))
    model.add(Dropout(0.4))
    #Conv 4
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    #Conv 5
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(996, activation='elu'))
    model.add(Dropout(0.5))

    model.add(Dense(100, activation='elu'))
    model.add(Dropout(0.5))

    model.add(Dense(50, activation='elu'))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='elu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))

    plot_model(model, to_file=output_dir + 'model.png', show_shapes=True)
    return model


# %% LOAD & COMPILE MODEL
# model = mgpu.make_parallel(model,2) #enable parallel gpu
model = make_model()
adam = optimizers.adam(lr=8e-4)
if (load_model):
    model.load_weights(load_model)
model.compile(loss='mse', optimizer=adam)

# %% RUN MODEL
model_file = 'model.{epoch:02d}-{val_loss:.4f}.h5'
model_checkpoint = ModelCheckpoint(
    output_dir + model_file, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, period=1)
early_stopping = EarlyStopping(
    monitor='val_loss', min_delta=1e-4, patience=30, verbose=0, mode='auto')
tensorboard = TensorBoard(
    log_dir= output_dir + 'logs', histogram_freq=1, write_graph=True, write_images=True, embeddings_freq=1)

train_generator = BatchGenerator(data, 'training', batch_size=256)
valid_generator = BatchGenerator(data, 'validation', batch_size=256)

history_object = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.steps,
    validation_data=valid_generator,
    validation_steps=valid_generator.steps,
    epochs=300,
    callbacks=[early_stopping, model_checkpoint, tensorboard],
    max_q_size=30,
    workers=6,
    pickle_safe=False)

# %% PLOT LOSS
plot_loss(history_object, to_file=output_dir + 'loss.png')

# TO START SERVER DO CALL
# python drive.py output/model.h5
