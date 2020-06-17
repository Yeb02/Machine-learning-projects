import requests
import numpy as np
import cv2
import datetime
import matplotlib.pyplot as plt
import pygame
import random


###### Network 1 adapted, uncommented for clarity, details in the original sourcecode.
class Network(object):
    def __init__(self,sizes): #the list sizes contains the number of neurons in the respective layers.
        self.num_layers = len(sizes)  #the number of the layers in Network
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x)
                        for x,y in zip(sizes[:-1],sizes[1:])]

    def feedforward(self,a):
        """Return the output of the network if "a" is input"""
        for b,w in zip(self.biases,self.weights):
            a = sigmoid(np.dot(w,a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size,eta,
            test_data = None):

        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)        #rearrange the training_data randomly
            mini_batches = [ training_data[k:k + mini_batch_size]
                                            for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self,mini_batch,eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                        for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        # print('trouvé', test_results[0][0])
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

import sys
sys.path.append(r"C:\Users\alpha\OneDrive\Bureau\Informatique\machine_learning\neural-networks-and-deep-learning-master\src")

import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

training_data = [[cv2.threshold((255*training_data[k][0]).astype('uint8'), 200, 255, cv2.THRESH_BINARY)[1], training_data[k][1]] for k in range(50000)]


test_data = [[cv2.threshold((255*training_data[k][0]).astype('uint8'), 200, 255, cv2.THRESH_BINARY)[1], test_data[k][1]] for k in range(10000)]


# net = Network([784, 30, 10])
net = Network([784, 30, 30, 10])

# net.SGD(training_data, 5, 11, 3.0, test_data=None)
net.SGD(training_data, 10, 50, 3.0, test_data=test_data)



##### Url fetch, zoom
url = 'http://192.168.43.1:8080/shot.jpg'     #prompt the device's temporary ipv4 adress

img_resp = requests.get(url)
img_arr = np.array(bytearray(img_resp.content), dtype = np.uint8)
img = cv2.imdecode(img_arr, -1)

zoom = 3
ne = np.int(img.shape[0]/zoom)
l1 = np.int(img.shape[1]/2 - (img.shape[0] - 2 * ne)/2)
l2 = np.int(img.shape[1]/2 + (img.shape[0] - 2 * ne)/2)

pygame.init()
screen = pygame.display.set_mode((500, 500))
pygame.display.set_caption("sampling window")
pygame.event.get()


def tester(img, expectation):
    data = [[np.expand_dims(np.reshape(img, 784), 1), expectation]]
    net.evaluate(data)


run = True
i = 0

##### recording or live loop
while run:
    pygame.event.get()
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype = np.uint8)
    img = cv2.imdecode(img_arr, -1)

    img_zoomed = img[ne:-ne, l1:l2]
    img_sized = cv2.resize(img_zoomed, (28, 28))
    gray = cv2.cvtColor(img_sized, cv2.COLOR_BGR2GRAY)
    graybis = 1 - gray/255
    (thresh, g2) = cv2.threshold((graybis * 255).astype('uint8'), 200, 255, cv2.THRESH_BINARY)

    keys = pygame.key.get_pressed()
    if keys[pygame.K_SPACE]:
        filename = r'C:\Users\alpha\OneDrive\Bureau\Informatique\machine_learning\Divers\images_enregistrees'
        filename += '\capture%06d.png' % i
        enreg = cv2.cvtColor((g2*255).astype('uint8'),cv2.COLOR_GRAY2RGB) #n'enregistre que le rgb...
        cv2.imwrite(filename, enreg)
        print('saved')
        i += 1

    if keys[pygame.K_ESCAPE]:
        run = False

    if keys[pygame.K_a]:   #touche q sur les claviers usuels.
        tester(g2, 0)

    plt.clf()
    plt.imshow(g2, cmap = 'gist_gray')
    plt.pause(.1)
    # pygame.time.delay(50)

pygame.quit()
plt.clf()
plt.show()
