import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

np.random.seed(0)


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

    return  coeff * np.exp(- dist / (2 * sigma_n ** 2)).sum(axis=-1)

def sig(As, dim, range):
    x = np.linspace(range[0], range[1], dim[0])
    y = np.linspace(range[0], range[1], dim[1])
    r_x, r_y = np.meshgrid(x, y)
    sigma_s = np.random.uniform(0, 1/2)
    c_x, c_y = np.random.uniform(-1/4, 1/4), np.random.uniform(-1/4, 1/4)
    rect_r = np.logical_and(rect(r_x), rect(r_y))

    singnal = As * rect_r * circ(r=(r_x, r_y), center_coords=(c_x, c_y), sigma_s=sigma_s)

    return singnal, sigma_s

def backgrd(sigma_n, a_n, N, dim, range):
    x = np.linspace(range[0], range[1], dim[0])
    y = np.linspace(range[0], range[1], dim[1])
    X, Y = np.meshgrid(x, y)
    
    rect_XY = np.logical_and(rect(X), rect(Y))
    lumps = a_n * gaussian(X, Y, sigma_n, N)
    
    return rect_XY * lumps


def save_figure(imgs, titles, save_path):
    fig = plt.figure()
    num = len(imgs)
    for i in range(num):
        plt.subplot(1, num, i+1)
        plt.imshow(imgs[i])
        plt.axis('off')
        plt.title(titles[i])

    plt.savefig(save_path)
    plt.close('all')


def generate_object(num_samples, s_range=(-1/2, 1/2), dim=(64, 64), out_dir="data", save_fig=True):
    try:
        os.mkdir(out_dir)
    except OSError as error:
        pass 

    for i in range(num_samples):
        signal, sigma_s = sig(3, dim=dim, range=(-1/2, 1/2))
        background = backgrd(sigma_s, 1, 10, dim=dim, range=(-1/2, 1/2))
        object = signal + background
        mask = np.ones(dim)
        mask[signal > 0] = 2
        data = np.stack([signal, background, object, mask])
        save_path = os.path.join(out_dir, str(i))

        try:
            os.mkdir(save_path)
        except OSError as error:
            pass

        with open(f"{save_path}/orig.npy", 'wb') as f:
            np.save(f, data)

        if save_fig:
            titles = ['signal', 'background', 'object', 'mask']
            save_figure(data, titles, f"{save_path}/vis.jpg")


        #return data
