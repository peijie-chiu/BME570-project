import numpy as np


def generate_sys_matrix(dim=64):
    flip_angles = ((np.pi / 8) * np.arange(0, 8)).reshape(-1, 1)
    m = (np.arange(1, 17, 1) - 8).reshape(1, -1)

    km_x = (np.cos(flip_angles) * m).reshape(-1, 1)
    km_y = (np.sin(flip_angles) * m).reshape(-1, 1)
    km = np.hstack([km_x, km_y])

    x = np.linspace(-1/2, 1/2, dim)
    y = np.linspace(-1/2, 1/2, dim)

    xx, yy = np.meshgrid(x, y)
    coords = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])

    H = np.exp(-2j * np.pi * km.dot(coords.T))
    
    return H

def svd(H):
    H_adjoint_H = H.T.dot(H)
    eigen_values, eigen_vectors = np.linalg.eig(H_adjoint_H)
    print(eigen_values)

H = generate_sys_matrix()
svd(H)