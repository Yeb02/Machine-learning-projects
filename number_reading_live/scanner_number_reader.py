import requests
import numpy as np
import cv2
import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pygame
import random
import sys


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
        if len(test_results) == 1:
            return test_results[0][0]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


if False: #Ne faire tourner qu'une fois par shell, mais ne pas oublier !

    sys.path.append(r"C:\Users\alpha\OneDrive\Bureau\Informatique\machine_learning\neural-networks-and-deep-learning-master\src")

    import mnist_loader
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    sys.path.append(r"C:\Users\alpha\OneDrive\Bureau\Informatique\machine_learning\number_reading_live")
    import dataset_rework
    training_data, validation_data, test_data = dataset_rework.generalise(training_data, validation_data, test_data)

    net = Network([784, 30, 10])
    # net = Network([784, 30, 30, 10])

    net.SGD(training_data, 12, 11, 3.0, test_data=test_data)  #attendre au moins l'epoch 10, l'efficacité bondit (81->92 %)
    # net.SGD(training_data, 10, 50, 3.0, test_data=test_data)

def tester(img, expectation):
    data = [[np.expand_dims(np.reshape(img, 784), 1), expectation]]
    return net.evaluate(data)

##### Url fetch, zoom, miscellaneous tests.

def check():
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

def show(im):
    im = np.array(im)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)   #pour tester.
    graybis = 1 - gray/255
    (thresh, im) = cv2.threshold((graybis * 255).astype('uint8'), 80, 255, cv2.THRESH_BINARY)
    plt.imshow(im)
    plt.show()



##### flooking for numbers through the image at different scales. image has to be significantly greater than 28*28

def research(im):  #on fait l'hypothèse que trouver le meme resultat dans une zone signifie qu'il y a effectivement un nombre. Le réseau est extremement rapide à évaluer, on peut se permettre ces calculs supplémentaires..
    im = np.array(im)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)   #pour tester.
    graybis = 1 - gray/255
    (thresh, im) = cv2.threshold((graybis * 255).astype('uint8'), 80, 255, cv2.THRESH_BINARY)


    co = [] #valeur, cox, coy, taille de l'immage
    echelle = 1 #nombre de sous divisions de l'image
    marge = 2 #zone explorée pour vérifier la robustesse du résultat
    m = min(im.shape[0], im.shape[1])
    while m/echelle > 28:
        w = m/echelle
        bornei = int(im.shape[0]/w) - 1
        bornej = int(im.shape[1]/w) - 1
        for i in range(bornei):
            for j in range(bornej):
                a = tester(np.resize(im[int(w*i) : int(w*(i + 1)), int(w*j) : int(w*(j + 1))], (28, 28)), 0)

                b = tester(np.resize(im[int(w*i) + marge: int(w*(i + 1)) + marge * (i != bornei - 1), int(w*j) - marge  * (j != 0): int(w*(j + 1)) - marge], (28, 28)), 0)

                c = tester(np.resize(im[int(w*i) + marge: int(w*(i + 1)) + marge * (i != bornei - 1), int(w*j) + marge: int(w*(j + 1)) + marge * (j != bornej - 1)], (28, 28)), 0)

                d = tester(np.resize(im[int(w*i) - marge * (i != 0): int(w*(i + 1)) - marge, int(w*j) - marge * (j != 0): int(w*(j + 1)) - marge], (28, 28)), 0)

                e = tester(np.resize(im[int(w*i) - marge  * (i != 0): int(w*(i + 1)) - marge, int(w*j) + marge: int(w*(j + 1)) + marge * (j != bornej - 1)], (28, 28)), 0)

                if a == b == c == d == e != 5:
                    co.append([tester(np.resize(im[int(w*i) : int(w*(i + 1)), int(w*j) : int(w*(j + 1))], (28, 28)), 0), int(w*i), int(w*j), int(w)])
                    # plt.imshow(np.resize(im[int(w*i) : int(w*(i + 1)), int(w*j) : int(w*(j + 1))], (28, 28)))
                    # print(int(w*i), int(w*(i + 1)), int(w*j), int(w*(j + 1)))
                    # plt.pause(2)


        echelle *= 2
    # plt.show()
    return co


def research_demo(im):
    co = research(im)
    fig,ax = plt.subplots(1)
    ax.imshow(im)
    for elt in co:

        rectangle = patches.Rectangle((elt[1], elt[2]),elt[3],elt[3],linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rectangle)
        plt.text(elt[1], elt[2], str(elt[0]))

    plt.show()



####### live loop

coglob = [] #liste des listes des chiffres trouvés
run = False
i = 0

while run:
    pygame.event.get()
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype = np.uint8)
    img = cv2.imdecode(img_arr, -1)

    # img_zoomed = img[ne:-ne, l1:l2]
    # img_sized = cv2.resize(img_zoomed, (28, 28))
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
        print(tester(g2, 0))
        coglob += research(g2)

    plt.clf()
    plt.imshow(g2, cmap = 'gist_gray')
    plt.pause(.1)
    # pygame.time.delay(50)

pygame.quit()
plt.clf()
plt.show()
