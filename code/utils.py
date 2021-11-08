import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['image.cmap'] = 'viridis'

def rect(x):
    return np.abs(x) < 0.5

def circ(r:tuple, center_coords:tuple, sigma_s):
    r_x, r_y = r
    c_x, c_y = center_coords
    return (r_x - c_x) ** 2 + (r_y-c_y) ** 2 <= sigma_s ** 2

def gaussian(X, Y, sigma_n, N):
    X, Y = X[..., np.newaxis], Y[..., np.newaxis]
    c_x, c_y = np.random.uniform(-1/2, 1/2, N), np.random.uniform(-1/2, 1/2, N)
    c_x, c_y = c_x.reshape(1, 1, -1), c_y.reshape(1, 1, -1)
    coeff = 1 / (2 * np.pi * sigma_n ** 2)
    dist = (X - c_x) ** 2 + (Y - c_y) ** 2
    dist = dist.sum(axis=-1)

    return  coeff * np.exp(- dist / (2 * sigma_n ** 2))

def sig(As, dim, range):
    x = np.linspace(range[0], range[1], dim[0])
    y = np.linspace(range[0], range[1], dim[1])
    r_x, r_y = np.meshgrid(x, y)
    sigma_s = np.random.uniform(1, 1/2)
    c_x, c_y = np.random.uniform(-1/4, 1/4), np.random.uniform(-1/4, 1/4)
    rect_r = np.logical_and(rect(r_x), rect(r_y))

    singnal = As * rect_r * circ(r=(r_x, r_y), center_coords=(c_x, c_y), sigma_s=sigma_s)

    return singnal

def backgrd(sigma_n, a_n, N, dim, range):
    x = np.linspace(range[0], range[1], dim[0])
    y = np.linspace(range[0], range[1], dim[1])
    X, Y = np.meshgrid(x, y)
    
    rect_XY = np.logical_and(rect(X), rect(Y))
    lumps = a_n * gaussian(X, Y, sigma_n, N)
    
    return rect_XY * lumps

signal = sig(3, dim=(64, 64), range=(-4, 4)) 
background = backgrd(1, 1, 10, dim=(64, 64), range=(-1/8, 1/8))
print(signal.shape, background.shape)
f = signal + background 
print(f)
plt.figure()

plt.subplot(1, 3, 1)
plt.imshow(signal)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(background)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(f)
plt.axis('off')

plt.show()

