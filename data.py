import csv
import os

import h5py
import keras.preprocessing.image
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import cv2


class Data:
    def __init__(self, input):
        self.input = input
        X_train_paths, X_valid_paths, y_train, y_valid = train_test_split(
            input.image_paths, input.measurments, test_size=0.2)
        self.train_generator = self.generator(X_train_paths, y_train)
        self.validation_generator = self.generator(X_valid_paths, y_valid)

    @classmethod
    def fromFile(self, src):
        file = src + 'data.h5'
        with h5py.File(file) as hf:
            image_paths = hf['image_paths'][:]
            measurments = hf['measurments'][:]
            return self(src, Input(src, image_paths, measurments))

    def save(self, src):
        file = src + 'data.h5'
        with h5py.File(file, 'w') as hf:
            hf.create_dataset("image_paths", data=self.input.image_paths)
            hf.create_dataset("measurments", data=self.input.measurments)

    def generator(self, X_paths, y, batch_size=32):
        num_samples = len(X_paths)
        while 1:
            X_paths, y = shuffle(X_paths, y)
            for offset in range(0, num_samples, batch_size):
                image_paths = X_paths[offset:offset + batch_size]
                angles = y[offset:offset + batch_size]

                images, angles = zip(*[
                    self.transform(i, a) for i, a in zip(image_paths, angles)
                ])

                # trim image to only see section with road
                X_batch = np.array(images)
                y_batch = np.array(angles)
                yield (X_batch, y_batch)

    def transform(self, image_path, angle):
        full_path = self.input.src + image_path.decode('UTF-8')
        image = cv2.imread(full_path)
        return (image, angle)


input_dir = 'input_v1/'
output_dir = 'output/'


class Input:
    def add_sample(self, image_src, measurment):
        # image = cv2.imread(image_src)
        image_np = np.string_(image_src)
        self.__add(image_np, measurment)
        #add flipped image
        # self.__add(np.fliplr(image), -measurment)

    def __add(self, image, measurment):
        self.image_paths.append(image)
        self.measurments.append(measurment)

    def __init__(self, src, image_paths=[], measurments=[]):
        self.src = src
        self.image_paths = image_paths
        self.measurments = measurments

    @classmethod
    def fromFile(self, dir_src):
        file = dir_src + 'driving_log.csv'
        print('reading input from ' + file)
        with open(file) as csv_file:
            new_input = Input(dir_src)
            reader = csv.reader(csv_file)
            for line in reader:
                steering_center = float(line[3])
                # create adjusted steering measurements for the side camera images
                correction = 0.25  # parameter to tune
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                image_center = line[0]
                new_input.add_sample(image_center, steering_center)

                image_left = line[1]
                new_input.add_sample(image_left, steering_left)

                image_right = line[2]
                new_input.add_sample(image_right, steering_right)
        return new_input


input = Input.fromFile(input_dir)
data = Data(input)
data.save(input_dir)

# data = Data.fromFile(input_dir)
tgen = data.train_generator
img, angle = tgen.__next__()
cv2.imshow('image', img[0])
cv2.waitKey(0)
cv2.destroyAllWindows()

# for img, angle in tgen:
#     cv2.imshow('image',img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     break
