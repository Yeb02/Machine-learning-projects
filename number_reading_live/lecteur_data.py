import numpy as np
import cv2
import matplotlib.pyplot as plt

for k in range(10000):
    plt.clf()
    plt.imshow(np.reshape(validation_data[k][0], (28,28)), cmap = 'gist_gray')
    print(net.evaluate([[validation_data[k][0], validation_data[k][1]]]))
    plt.pause(2)
plt.show()

net.evaluate([[np.expand_dims(np.reshape(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY), 784), 1), 0]]), plt.imshow(np.reshape(i, (28,28,3)), cmap = 'gist_gray'), plt.show()

net.evaluate([[t, 0]]), plt.imshow(np.reshape(t, (28,28)), cmap = 'gist_gray'), plt.show()