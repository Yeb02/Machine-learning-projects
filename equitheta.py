import cv2, time, sys, os
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

a = array([0, 0, 0])
b = array([1, 0, 0])
c = array([0, 2, 0])

theta_ab = 0.6154797086703871
theta_bc = 1.5707963267948966
theta_ca = 1.2309594173407747
n = 50
tr = .005

def theta(m, u, v):
    um_carre = sum((m - u) ** 2)
    vm_carre = sum((m - v) ** 2)
    uv_carre = sum((u - v) ** 2)
    return np.arccos(( um_carre + vm_carre - uv_carre) / (2 * np.sqrt(um_carre * vm_carre)) )


def deux_points(i):
    X = np.linspace(-3/2, 5/2, i)
    Y = np.linspace(-2, 2, i)
    Z = np.linspace(-2, 2, i)

    mx, my, mz = [0, 1], [0, 0], [0, 0]
    ax.scatter(mx, my, mz, 'ro', s=1030)

    for x in X:
        for y in Y:
            for z in Z:
                if abs(theta(array([x, y, z]), a, b)-theta_ab) < .05:
                    mx.append(x)
                    my.append(y)
                    mz.append(z)
    return(mx, my, mz)

def trois_points(i):
    X = np.linspace(-1/2, 3/2, i)
    Y = np.linspace(0, 2, i)
    Z = np.linspace(-1, 1, i)

    mx, my, mz = [0, 1, 0], [0, 0, 2], [0, 0, 0]
    ax.scatter(mx, my, mz, 'ro', s=500)

    for x in X:
        for y in Y:
            for z in Z:
                if abs(theta(array([x, y, z]), a, b)-theta_ab) < tr and abs(theta(array([x, y, z]), b, c)-theta_bc) < tr and abs(theta(array([x, y, z]), c, a)-theta_ca) < tr:
                    mx.append(x)
                    my.append(y)
                    mz.append(z)
    return(mx, my, mz)

# mx, my, mz = deux_points(n)
mx, my, mz = trois_points(n)

ax.scatter(mx, my, mz)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

lim = 1
ax.set_zlim3d(-lim, lim)
ax.set_ylim3d(-lim, lim)
ax.set_xlim3d(-lim, lim)

plt.show()