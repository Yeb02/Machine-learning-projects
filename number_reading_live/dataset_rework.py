import sys, os
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt

sys.path.append(r'C:\Users\alpha\OneDrive\Bureau\Informatique\machine_learning\neural-networks-and-deep-learning-master\src')

import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# transorming the dataset in order to use the network for diverse pictures, converted to black and white and flipped or sideways (classic data augmentation)

def generalise(training_data, validation_data, test_data):

    tr2 = []
    for k in range(len(training_data)):

        (tresh, q) = cv2.threshold((training_data[k][0] * 255).astype('uint8'), 200, 255, cv2.THRESH_BINARY)

        # plt.clf()    #visualisation
        # plt.subplot(221)
        # plt.imshow(np.reshape(training_data[k][0], (28, 28)))

        training_data[k] = list(training_data[k])
        training_data[k][0] = q/255
        tr2.append([np.reshape(np.transpose(np.reshape(training_data[k][0], (28,28))),(784, 1)), training_data[k][1]])

        # plt.subplot(223)
        # plt.imshow(np.reshape(tr2[-1][0], (28, 28)))
        # plt.subplot(222)
        # plt.imshow(np.reshape(training_data[k][0], (28, 28)))
        # plt.pause(2)

    plt.show()

    for k in range(len(test_data)):
        (tresh, q) = cv2.threshold((test_data[k][0] * 255).astype('uint8'), 200, 255, cv2.THRESH_BINARY)
        test_data[k] = list(test_data[k])
        test_data[k][0] = q/255

    for k in range(len(validation_data)):
        (tresh, q) = cv2.threshold((validation_data[k][0] * 255).astype('uint8'), 200, 255, cv2.THRESH_BINARY)
        validation_data[k] = list(validation_data[k])
        validation_data[k][0] = q/255

    return training_data, validation_data, test_data




