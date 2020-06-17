import sys
import cv2
import numpy as np
from sympy import sin, cos, Matrix
from sympy.abc import rho, phi
import os
import requests
import datetime
import time
import sympy
from sympy import *
import random
sift = cv2.xfeatures2d.SIFT_create()

# http://fourier.eng.hmc.edu/e176/lectures/NM/node21.html

###### Acquisition, désactiver en fin de paragraphe si déjà fait, à adapter si chgt de machine/location.

path  = r'C:\Users\alpha\OneDrive\Bureau\Informatique\pix4D\takes\take_2\\'
url = 'http://100.85.234.124:8080/shot.jpg'

def take(url):
    compteur = 0
    while True:
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype = np.uint8)
        img = cv2.imdecode(img_arr, -1)
        # img_sized = cv2.resize(img, (800, 600))
        cv2.imshow('stream', img)

        if cv2.waitKey(1) == 27:   #escape, presser avant de fermer la fenetre du live
            break

        if cv2.waitKey(1) == 101:   #touche e comme enregistrer, maintenir pendant une seconde pour capturer
            filename = path
            filename += 'pic_' + str(compteur) + '.jpg'
            print(filename)
            cv2.imwrite(filename, img)
            print('saved')
            compteur += 1
    print(compteur)


# take(url)

###### Capture par drone, paramétrer son trajet à l'avance (en gardant le controle !)

######  Ouverture du repertoire, à adapter si chgt de machine/location.

directory = os.fsencode(r'C:\Users\alpha\OneDrive\Bureau\Informatique\pix4D\takes\take_2')
repertoire = []

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".png") or filename.endswith(".jpg"):
        path2 = path  + str(filename)
        repertoire.append(cv2.imread(path2))  #denoise ?
    else:
        continue

######  Init du sift

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params,search_params)

###### Fonction de localisation qui place l'image k+1 et le nuage de points par rapport à la k-ieme et à l'objet, avec la méthode de Newton Raphson:   http://fourier.eng.hmc.edu/e176/lectures/NM/node21.html

def Newton_Raphson(points, indices, u, v, x1, y1, vecteurs):  # utiliser des pts des images d'avant pour obtenir la constante de distance.
    t1 = []

    if len(indices) != 0:  #○ faire les minis-batches  ♀♂
        for i in indices:
            t1 += [vecteurs[-1][i]]
        l = len(indices)
    else:
        t1 += [1]
        l = 1

    t1 += [random.random() for k in range(n - l)]
    t2 = [random.random() for k in range(n)]  # variables d'intersection, à priori positives.
    phi, theta, psi = 0, 0, 0 #angles d'euler de cam2 dans la base de cam1, l' objectif regarde les +z
    X2, Y2, Z2 = 0, 0, 0 #coo de cam 2 par rapport à cam 1 dans la base de cam1

    def descente_de_gradient(Xn0, x1, y1, u, v, l, t1):
        Xn = sympy.IndexedBase('Xn')
        J = Jacobien(Xn, x1, y1, u, v, l, t1)
        epsilon = .5
        Xn1 = [0] * (2*n + 6)  #doit etre (tres ?) different de Xn0.
        cpt = 0
        while sum(abs(abs(np.array(Xn0)) - abs(np.array(Xn1)))) > epsilon and cpt < 20:
            # print(Xn0, len(Xn0), type(Xn0), sum(abs(abs(np.array(Xn0)) - abs(np.array(Xn1)))))
            print(sum(abs(abs(np.array(Xn0)) - abs(np.array(Xn1)))))
            if cpt != 0:
                Xn0 = list(Xn1)
            J1 = J.subs([(Xn[i], Xn0[i]) for i in range(l, 2*n + 6, 1)])
            taille = J1.shape
            J2 = np.zeros(taille)
            for i in range(taille[0]):
                for j in range(taille[1]):
                    J2[i,j]=sympy.N(J1[i,j])
            J2 = np.matrix(J2)
            J3 = (((J2.T).dot(J2)).I).dot(J2.T)
            fXn = f(Xn, x1, y1, u, v, l, t1)
            fXn = [fXn[j].subs([(Xn[i], Xn0[i]) for i in range(l, 2*n + 6, 1)]) for j in range(len(fXn))]
            Xn1 = t1[:l] + list(np.array(Xn0[l::] - J3.dot(fXn))[0])
            cpt += 1
        return(Xn0)

    Xn0 = t1 + t2
    Xn0 += [phi, theta, psi, X2, Y2, Z2]

    return descente_de_gradient(Xn0, x1, y1, u, v, l, t1)


###### f et son jacobien, rendre la taille du batch variable ?


def f(Xn, x1, y1, u, v, l, t1):
    n = len(x1)
    fct = [0] * 3 * n


    for i in range(l): #Vecteur colonne de f

        fct[3*i] = t1[i]*x1[i] - Xn[2*n + 3] - Xn[n + i]*( u[i] * (cos(Xn[2*n +2])*cos(Xn[2*n]) - sin(Xn[2*n +2])*cos(Xn[2*n +1])*sin(Xn[2*n]) ) + v[i] * (-cos(Xn[2*n +2])*sin(Xn[2*n]) - sin(Xn[2*n +2])*cos(Xn[2*n +1])*cos(Xn[2*n]) ) + sin(Xn[2*n +2])*sin(Xn[2*n + 1])  )

        fct[3*i + 1] = t1[i]*y1[i] - Xn[2*n + 4] - Xn[n + i]*( u[i] * (sin(Xn[2*n +2])*cos(Xn[2*n]) + cos(Xn[2*n +2])*cos(Xn[2*n +1])*sin(Xn[2*n]) ) + v[i] * (-sin(Xn[2*n +2])*sin(Xn[2*n]) + cos(Xn[2*n +2])*cos(Xn[2*n +1])*cos(Xn[2*n]) ) - cos(Xn[2*n +2])*sin(Xn[2*n +1])  )

        fct[3*i + 2] = t1[i] - Xn[2*n + 5] - Xn[n + i]*( u[i] * sin(Xn[2*n +1])*sin(Xn[2*n]) + v[i] * sin(Xn[2*n +1]) * cos(Xn[2*n]) + cos(Xn[2*n +1]) )


    for i in range(l, n, 1): #Vecteur colonne de f

        fct[3*i] = Xn[i]*x1[i] - Xn[2*n + 3] - Xn[n + i]*( u[i] * (cos(Xn[2*n +2])*cos(Xn[2*n]) - sin(Xn[2*n +2])*cos(Xn[2*n +1])*sin(Xn[2*n]) ) + v[i] * (-cos(Xn[2*n +2])*sin(Xn[2*n]) - sin(Xn[2*n +2])*cos(Xn[2*n +1])*cos(Xn[2*n]) ) + sin(Xn[2*n +2])*sin(Xn[2*n + 1])  )

        fct[3*i + 1] = Xn[i]*y1[i] - Xn[2*n + 4] - Xn[n + i]*( u[i] * (sin(Xn[2*n +2])*cos(Xn[2*n]) + cos(Xn[2*n +2])*cos(Xn[2*n +1])*sin(Xn[2*n]) ) + v[i] * (-sin(Xn[2*n +2])*sin(Xn[2*n]) + cos(Xn[2*n +2])*cos(Xn[2*n +1])*cos(Xn[2*n]) ) - cos(Xn[2*n +2])*sin(Xn[2*n +1])  )

        fct[3*i + 2] = Xn[i] - Xn[2*n + 5] - Xn[n + i]*( u[i] * sin(Xn[2*n +1])*sin(Xn[2*n]) + v[i] * sin(Xn[2*n +1]) * cos(Xn[2*n]) + cos(Xn[2*n +1]) )
    return fct


def Jacobien(Xn, x1, y1, u, v, l, t1):  #Vrai_Xn : (t11,t21,t12,t22...t2n,phi,theta,psi,X2,Y2,Z2) en valeurs numériques
    n = len(x1)
    fct = f(Xn, x1, y1, u, v, l, t1)
    F = Matrix([fct[i] for i in range(3*n)])
    X_n = Matrix([Xn[i] for i in range(l, 2*n + 6, 1)])
    return F.jacobian(X_n)


######  Loop sur le repertoire

ouv = 1  #sur le drone, et en radians. Calculer sur le téléphone.
lrep = len(repertoire)
t0 = time.time()
sh = repertoire[0].shape
largeur_en_pixels = sh[1]
hauteur_en_pixels = sh[0]
points = []
vecteurs = []

kp1, des1 = sift.detectAndCompute(repertoire[0],None)

for k in range(lrep - 1):
    t = time.time()
    pts1, pts2 = [], []
    kp2, des2 = sift.detectAndCompute(repertoire[k + 1],None)
    matches = flann.knnMatch(des1,des2,k=2)

    for i,(m,n) in enumerate(matches):
        if m.distance < 0.4*n.distance:  #.8, minimum 6 points retenus, cela confirme la théorie.
            pts2.append([int(kp2[m.trainIdx].pt[1]), int(kp2[m.trainIdx].pt[0])])
            pts1.append([int(kp1[m.queryIdx].pt[1]), int(kp1[m.queryIdx].pt[0])])

    kp1, des1 = kp2, des2

    indices = []
    if k != 0:
        pts_prime1 = []
        pts_reserve1 = []
        pts_prime2 = []
        pts_reserve2 = []
        indices = []
        for (j, elt1) in enumerate(pts1):
            val = 0
            for (i, elt2) in enumerate(points[-1]):
                if elt1 == elt2:
                    pts_prime1.append(elt1)
                    pts_prime2.append(pts2[j])
                    indices.append(i)
                    val = 1
            if val == 0:
                pts_reserve1.append(elt1)
                pts_reserve2.append(pts2[j])
        points += [pts_prime1 + pts_reserve1[:len(indices)], pts_prime2 + pts_reserve2[:len(indices)]]

    else:
        points += [pts1, pts2]


    print('nb de points retenus:', len(points[-1]))
    print('nb de similarités:', len(indices))


    pts1 = points[-2]
    pts2 = points[-1]
    n = len(pts1)
    u = []
    v = []
    x1 = []
    y1 = []
    tng = np.tan(ouv/2)
    l = largeur_en_pixels/2
    h = hauteur_en_pixels

    for i in range(n):
        Xpix1 = l - pts1[i][1]
        Ypix1 = h - pts1[i][0]
        Xpix2 = l - pts2[i][1]
        Ypix2 = h - pts2[i][0]
        u.append(Xpix2 * tng /l)
        v.append(Ypix2 * tng /l)
        x1.append(Xpix1 * tng /l)
        y1.append(Ypix1 * tng /l)



    vecteurs += [Newton_Raphson(points, indices, u, v, x1, y1, vecteurs)]

    print(k + 1, 'done out of', lrep - 1, 'in', round(time.time() - t0, 2), '(', round(time.time() - t, 2), ')')