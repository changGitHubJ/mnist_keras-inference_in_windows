from __future__ import print_function
import keras
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import InputLayer, Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import multi_gpu_model

gpu_count=2

class Model:
    model = 0
    batch_size = 0
    num_classes = 0
    epochs = 0

    def __init__(self, batch_size, num_classes, epochs):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.epochs = epochs

    def create_model(self, input_shape, multi_gpu=False):
        if not multi_gpu:
            self.model = Sequential()
            self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
            self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Dropout(0.25))
            self.model.add(Flatten())
            self.model.add(Dense(128, activation='relu'))
            self.model.add(Activation('relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(self.num_classes, activation='softmax'))
        else:
            with tf.device("/cpu:0"):
                model = Sequential()
                model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
                model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Dropout(0.25))
                model.add(Flatten())
                model.add(Dense(128, activation='relu'))
                model.add(Activation('relu'))
                model.add(Dropout(0.5))
                model.add(Dense(self.num_classes, activation='softmax'))
            self.model = multi_gpu_model(model, gpus=gpu_count)
        self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

    def training(self, x_train, y_train, x_test, y_test):
        self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1, validation_data=(x_test, y_test))
    
    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test, verbose=0)

    def save(self, output_filename):
        self.model.save(output_filename)