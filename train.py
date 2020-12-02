import numpy as np
import os
# import random
# import tensorflow as tf
import time

import model
import load_data as data

# Architecture
n_hidden_1 = 256
n_hidden_2 = 256

# Parameter
learning_rate = 0.0001
training_epochs = 200
batch_size = 2048
display_step = 1
TRAIN_DATA_SIZE = 10000
VALID_DATA_SIZE = 500
TEST_DATA_SIZE = 500
IMG_SIZE = 28
OUTPUT_SIZE = 10
FILTER_SIZE_1 = 32
FILTER_SIZE_2 = 64

if __name__=='__main__':

    data = data.MyLoadData(IMG_SIZE, OUTPUT_SIZE)
    x_train = data.read_images('./data/trainImage.txt', TRAIN_DATA_SIZE)
    x_valid = data.read_images('./data/validationImage.txt', VALID_DATA_SIZE)
    x_test = data.read_images('./data/testImage.txt', TEST_DATA_SIZE)
    y_train = data.read_labels('./data/trainLABEL.txt', TRAIN_DATA_SIZE)
    y_valid = data.read_labels('./data/validationLABEL.txt', VALID_DATA_SIZE)
    y_test = data.read_labels('./data/testLABEL.txt', TEST_DATA_SIZE)
    
    input_shape = (IMG_SIZE, IMG_SIZE, 1)

    model = model.Model(batch_size, OUTPUT_SIZE, training_epochs)
    model.create_model(input_shape, multi_gpu=True)
    start = time.time()
    model.training(x_train, y_train, x_valid, y_valid)
    elapsed_time = time.time() - start
    print("time = %f"%elapsed_time)
    model.evaluate(x_test, y_test)
    model.save('./model.h5')
