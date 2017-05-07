import csv
import os

import h5py
from keras.preprocessing.image import load_img, img_to_array, flip_axis, random_shear, apply_transform, transform_matrix_offset_center
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
from PIL import ImageEnhance
from matplotlib import pyplot as plt


class Input:
    def add_sample(self, image_src, measurment):
        image_np = np.string_(image_src)
        self.__add(image_np, measurment)

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
                correction = 0.25
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                image_center = line[0]
                new_input.add_sample(image_center, steering_center)

                image_left = line[1]
                new_input.add_sample(image_left, steering_left)

                image_right = line[2]
                new_input.add_sample(image_right, steering_right)
        return new_input


class Data:
    def __init__(self, input):
        self.input = input
        X_train_paths, X_valid_paths, y_train, y_valid = train_test_split(
            input.image_paths, input.measurments, test_size=0.2)
        self.n_train_samples = len(X_train_paths)
        self.n_valid_samples = len(X_valid_paths)
        self.train_generator = self.generator(X_train_paths, y_train)
        self.validation_generator = self.generator(X_valid_paths, y_valid)

    @classmethod
    def fromFile(self, src):
        file = src + 'data.h5'
        with h5py.File(file) as hf:
            image_paths = hf['image_paths'][:]
            measurments = hf['measurments'][:]
            return self(Input(src, image_paths, measurments))

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

                X_batch = np.array(images)
                y_batch = np.array(angles)
                yield (X_batch, y_batch)

    def transform(self, image_path, angle):
        full_path = self.input.src + image_path.decode('UTF-8')
        img = load_img(full_path)

        # brightness
        intensity = np.random.uniform(low=0.25, high=1.35)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(intensity)

        img = np.array(img)

        # flip image
        if np.random.random() < 0.5:
            img = flip_axis(img, 1)
            angle = -angle

        transform_matrix = None

        # spatial shear
        intensity = 0.1
        shear = np.random.uniform(-intensity, intensity)
        shear_matrix = np.array([[1, -np.sin(shear), 0], [0, np.cos(shear), 0],
                                 [0, 0, 1]])
        transform_matrix = shear_matrix

        # random shift
        w_range = 0.15
        h_range = 0.15
        shift_matrix, angle = self.random_shift_matrix(img, w_range, h_range,
                                                       angle)
        transform_matrix = np.dot(transform_matrix, shift_matrix)

        # perform transformation
        img = self.transform_img(transform_matrix, img)

        # crop
        h, w = img.shape[0], img.shape[1]
        img = img[20:-32, 0:w]

        return (img, angle)

    def transform_img(self, transform_matrix, img):
        if transform_matrix is not None:
            h, w = img.shape[0], img.shape[1]
            transform_matrix = transform_matrix_offset_center(
                transform_matrix, h, w)
            img = apply_transform(
                img, transform_matrix, 2, fill_mode='nearest')
        return img

    def random_shift_matrix(self,
                            img,
                            w_range,
                            h_range,
                            angle,
                            pix_to_angle=0.004):
        h, w = img.shape[0], img.shape[1]
        tx = np.random.uniform(-h_range, h_range) * h
        ty = np.random.uniform(-w_range, w_range) * w
        transform_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])

        angle = angle - ty * pix_to_angle
        return transform_matrix, angle


def save_plot(images, angles, path):
    fig = plt.figure(figsize=(16, 8))
    for i in range(15):
        image = images[i]
        angle = angles[i]
        plt.subplot(3, 5, i + 1)
        plt.imshow(image)
        plt.axis('off')
        plt.title(str(np.round(angle, 2)))
    plt.savefig(path)


def plot_example_batch(plot_name='example',
                       input_dir='input_v1/',
                       output_dir='output/'):
    data = Data.fromFile(input_dir)
    tgen = data.train_generator

    imges, angles = tgen.__next__()
    save_plot(imges, angles, output_dir + plot_name)
