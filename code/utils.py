import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

def save_figure(imgs, titles, save_path, suptitle=None):
    plt.figure()
    if suptitle:
        plt.suptitle(suptitle)
    num = len(imgs)
    for i in range(num):
        plt.subplot(1, num, i+1)
        plt.imshow(imgs[i])
        plt.axis('off')
        plt.title(titles[i])
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close('all')

def rect(x):
    return np.abs(x) < 0.5

def pixelated(r):
    return np.logical_and(rect(r[0]), rect(r[1]))

def circ(r:tuple, center_coords:tuple, sigma_s):
    r_x, r_y = r
    c_x, c_y = center_coords
    return (r_x - c_x) ** 2 + (r_y-c_y) ** 2 <= sigma_s ** 2

def multi_sig(As, dim= (64, 64), s_range=(-1/2, 1/2), num=500):
    x = np.linspace(s_range[0], s_range[1], dim[0])
    y = np.linspace(s_range[0], s_range[1], dim[1])
    r_x, r_y = np.meshgrid(x, y)
    r_x, r_y = r_x[..., np.newaxis], r_y[..., np.newaxis]
    sigma_s = np.random.uniform(0, 1/2, size=(num, ))
    c_x, c_y = np.random.uniform(-1/4, 1/4, size=(1, 1, num)), np.random.uniform(-1/4, 1/4, size=(1, 1, num))
    signal = circ(r=(r_x, r_y), center_coords=(c_x, c_y), sigma_s=sigma_s)
    pixel = pixelated(r=(r_x, r_y))
    signal = As * pixel * signal

    return signal, sigma_s

def multi_gaussian(r, sigma_n, N=10, num=500):
    sigma_n = sigma_n[np.newaxis, np.newaxis, np.newaxis, ...]
    r_x, r_y = r
    r_x, r_y = r_x[..., np.newaxis], r_y[..., np.newaxis]
    c_x, c_y = np.random.uniform(-1/2, 1/2, size=(1, 1, N, num)), np.random.uniform(-1/2, 1/2, size=(1, 1, N, num))
    coeff = 1 / (2 * np.pi * sigma_n ** 2)
    dist = (r_x- c_x) ** 2 + (r_y-c_y) ** 2 
    lumps = coeff * np.exp(- dist / (2 * sigma_n ** 2))
    
    return lumps.sum(-2)
    
def multi_backgrd(sigma_n, a_n=1, N=10, dim=(64, 64), b_range=(-1/2, 1/2), num=500):
    x = np.linspace(b_range[0], b_range[1], dim[0])
    y = np.linspace(b_range[0], b_range[1], dim[1])
    r_x, r_y = np.meshgrid(x, y)
    r_x, r_y = r_x[..., np.newaxis], r_y[..., np.newaxis]
    pixel = pixelated(r=(r_x, r_y))
    lumps = a_n * multi_gaussian(r=(r_x, r_y), sigma_n=sigma_n, N=N, num=num)
    
    return pixel * lumps

# def gaussian(X, Y, sigma_n, N):
#     X, Y = X[..., np.newaxis], Y[..., np.newaxis]
#     c_x, c_y = np.random.uniform(-1/2, 1/2, N), np.random.uniform(-1/2, 1/2, N)
#     c_x, c_y = c_x.reshape(1, 1, -1), c_y.reshape(1, 1, -1)
#     coeff = 1 / (2 * np.pi * sigma_n ** 2)
#     dist = (X - c_x) ** 2 + (Y - c_y) ** 2

#     return  coeff * np.exp(- dist / (2 * sigma_n ** 2)).sum(axis=-1)

# def sig(As, dim, range):
#     x = np.linspace(range[0], range[1], dim[0])
#     y = np.linspace(range[0], range[1], dim[1])
#     r_x, r_y = np.meshgrid(x, y)
#     sigma_s = np.random.uniform(0, 1/2)
#     c_x, c_y = np.random.uniform(-1/4, 1/4), np.random.uniform(-1/4, 1/4)
#     rect_r = np.logical_and(rect(r_x), rect(r_y))

#     singnal = As * rect_r * circ(r=(r_x, r_y), center_coords=(c_x, c_y), sigma_s=sigma_s)

#     return singnal, sigma_s

# def backgrd(sigma_n, a_n, N, dim, range):
#     x = np.linspace(range[0], range[1], dim[0])
#     y = np.linspace(range[0], range[1], dim[1])
#     r_x, r_y = np.meshgrid(x, y)
    
#     rect_r = np.logical_and(rect(r_x), rect(r_y))
#     lumps = a_n * gaussian(r_x, r_y, sigma_n, N)
    
#     return rect_r * lumps



