import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from matplotlib import cm

import model
import load_data as data

IMG_SIZE = 28
OUTPUT_SIZE = 10
TEST_DATA_SIZE = 500
    
if __name__=='__main__':    
    with tf.device("/gpu:0"):

        data = data.MyLoadData(IMG_SIZE, OUTPUT_SIZE)
        x_test = data.read_images('./data/testImage.txt', TEST_DATA_SIZE)

        model = keras.models.load_model('model.h5')
        for i in range(TEST_DATA_SIZE):    
            ret = model.predict(x_test[i:i + 1], batch_size=1)   # OK
            print("predict ret:", ret)

            plt.subplot(1, 2, 1)
            plt.imshow(x_test[i].reshape(28, 28), cmap=cm.gray_r)
            plt.subplot(1, 2, 2)
            plt.plot(ret.reshape(10), np.linspace(0, 9, 10))
            plt.xlim([0, 1])
            plt.ylim([0, 10])
            plt.show()