import os
import numpy as np
# import matplotlib
# matplotlib.use('agg')
# import matplotlib.pyplot as plt
from utils import save_figure

def generate_matrix(object, mask, dim):
    # assert object.shape == (64, 64)
    delta_a = 1/64 * 1/64
    times = int(64 / dim)

    flip_angles = ((np.pi / 8) * np.arange(0, 8)).reshape(-1, 1)
    m = (np.arange(1, 17, 1) - 8).reshape(1, -1)

    km_x = (np.cos(flip_angles) * m).reshape(-1, 1)
    km_y = (np.sin(flip_angles) * m).reshape(-1, 1)
    # km = np.hstack([km_x, km_y])
    # print(km)

    x = np.linspace(-1 / 2, 1 / 2, 65)
    x_i = (x[0:64] + x[1:]) / 2
    y = np.linspace(-1 / 2, 1 / 2, 65)
    y_i = (y[0:64] + y[1:]) / 2
    X_i, Y_i = np.meshgrid(x_i, y_i)

    h_mn = np.zeros((16 * 8, dim * dim), dtype=complex)
    theta_n = np.zeros((dim * dim, 1))
    curmask = np.zeros((dim * dim, 1))
    for i in range(0, dim):
        for j in range(0, dim):
            km_ri = km_x.reshape(-1, 1) * X_i[i*times: (i+1)*times, j*times: (j+1)*times].reshape(1, -1) + km_y.reshape(-1, 1) * Y_i[i*times: (i+1)*times, j*times: (j+1)*times].reshape(1, -1)
            h_mn[:, i * dim + j] = np.sum(np.exp(-2 * np.pi * 1j * km_ri) * delta_a, axis=1)
            theta_n[i * dim + j, 0] = np.max(object[i*times: (i+1)*times, j*times: (j+1)*times])
            curmask[i * dim + j, 0] = np.max(mask[i*times: (i+1)*times, j*times: (j+1)*times])
    return h_mn, theta_n, curmask

def svd(h_mn):
    h_adjoint_h = h_mn.T.dot(h_mn)
    mu, u = np.linalg.eig(h_adjoint_h)

    index = np.argsort(mu)[::-1]
    mu = mu[index]
    u = u[:, index]

    num = np.min(h_mn.shape)
    v = h_mn.dot(u)
    v = v[:, 0:num] / np.sqrt(mu[0:num])

    mu = mu[0:num]
    return mu, v, u

def pseudo_inv(mu, v, u):
    m = v.shape[0]
    n = u.shape[0]
    k = mu.shape[0]
    h_mp = np.zeros((n, m))
    zero_appro = pow(10, -15)
    for i in range(k):
        if(mu[i].real < zero_appro):
            h_mp = h_mp + u[:, i].reshape(-1, 1) * v[:, i] / np.sqrt(mu[i])
    return h_mp

def generate_proj(theta_n, h_mn):
    proj = h_mn.dot(theta_n)
    return proj

def reconstruct(proj, h_mp, dim):
    recon = h_mp.dot(proj)
    recon = abs(recon)
    # print(recon)
    return recon

def generate_recon(num_objects, dir="data", orig_title="orig.npy"):
    dims = np.array([8, 16, 32])

    for dim in dims:
        for i in range(num_objects):
            data = np.load(dir+"/"+str(i)+"/"+orig_title)
            object = data[2]
            mask = data[3]

            h_mn, theta_n, curmask = generate_matrix(object, mask, dim)
            mu, v, u = svd(h_mn)
            h_mp = pseudo_inv(mu, v, u)
            ds_img = theta_n.reshape((dim, dim))
            ds_mask = curmask.reshape((dim, dim))

            proj = generate_proj(theta_n, h_mn)

            recon = reconstruct(proj, h_mp, dim)
            recon_img = recon.reshape((dim, dim))

            data = np.stack([ds_img, recon_img, ds_mask])

            save_path = os.path.join(dir, str(i))
            save_path = os.path.join(save_path, str(dim))

            try:
                os.mkdir(save_path)
            except OSError as error:
                pass

            with open(f"{save_path}/downsample.npy", 'wb') as f:
                np.save(f, data)

            titles = ['downsample_img', 'reconstruction', 'downsample_mask']
            save_figure(data, titles, f"{save_path}/vis.jpg")