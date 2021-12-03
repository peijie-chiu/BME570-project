import numpy as np
from skimage.io import imsave
# import matplotlib
# matplotlib.use('agg')
# import matplotlib.pyplot as plt

def downsample(obj, mask, factor=2.):
    image = obj.copy()
    mask_copy = mask.copy()
    if len(image.shape) == 2:
        image = image[np.newaxis, ...]
        mask_copy = mask_copy[np.newaxis, ...]

    B, H, W = image.shape 
    image_reshaped = image.reshape(B, H // factor, factor, W // factor, factor)
    mask_reshaped = image.reshape(B, H // factor, factor, W // factor, factor)

    return image_reshaped.max(axis=(2, 4)), mask_reshaped.max(axis=(2, 4))

def im2cols(k_size, stride, out_size):
    k_size, out_size = int(k_size), int(out_size)
    i0 = np.repeat(np.arange(k_size), k_size)
    i1 = stride * np.repeat(np.arange(out_size), out_size)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)

    j0 = np.tile(np.arange(k_size), k_size)
    j1 = stride * np.tile(np.arange(out_size), out_size)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    return i, j

def generate_sys_matrix(s_range=(-1/2, 1/2), dim=64, projection=(16, 8), downsample=1):
    delta = 1 / (dim ** 2)
    flip_angles = (np.pi / projection[1]) * np.arange(0, projection[1])
    m = (np.arange(1, projection[0] + 1, 1) - 8).reshape(-1, 1)
    km_x = (np.cos(flip_angles) * m).reshape(-1, 1)
    km_y = (np.sin(flip_angles) * m).reshape(-1, 1)
    km = np.hstack([km_x, km_y])

    x = np.linspace(s_range[0], s_range[1], dim)
    y = np.linspace(s_range[0], s_range[1], dim)

    r_x, r_y = np.meshgrid(x, y)
    coords = np.hstack([r_x.reshape(-1, 1), r_y.reshape(-1, 1)])

    H = np.exp(-2j * np.pi * km.dot(coords.T) / dim) 
    H = H * delta

    if downsample > 1:
        H = H.reshape(projection[0]*projection[1], dim, dim)
        i, j = im2cols(downsample, downsample, dim // downsample)
        H = H[:, i, j].sum(axis=1)

    return H

def reconstruction(obj, H):
    image = obj.copy()
    if len(image.shape) == 2:
        image = image[np.newaxis, ...]

    image = image.reshape(image.shape[0], -1)
    proj = H.dot(image.T)
    H_MP = np.linalg.pinv(H)
    recon = H_MP.dot(proj).T

    return np.abs(proj).T, recon


# H = generate_sys_matrix()
# H_reshape = H.reshape(128, 64, 64)
# i, j = im2cols(2, 2, 32)
# H_downsampled = H_reshape[:, i, j].sum(axis=1)

# with open(f"../data/new_raw/raw.npy", "rb") as f:
#     data = np.load(f)

# sample = data[0, ..., 0]
# # downsampled = sample[i, j].sum(axis=0).reshape(32, 32) / 4
# # print(downsampled.shape)
# downsampled = sample.reshape(32, 2, 32, 2)
# downsampled = downsampled.max(axis=(1, 3))
# plt.imshow(downsampled)
# plt.savefig("down_test.jpg")

# H_downsampled_MP = np.linalg.pinv(H_downsampled)
# downsampled_proj = H_downsampled.dot(downsampled.flatten())
# downsampled_recon = H_downsampled_MP.dot(downsampled_proj).reshape(32, 32)

# H_MP = np.linalg.pinv(H)
# proj = H.dot(sample.flatten())
# recon = H_MP.dot(proj).reshape(64, 64)

# plt.figure()
# plt.subplot(2, 3, 1)
# plt.imshow(sample, cmap='gray')
# plt.subplot(2, 3, 2)
# plt.imshow(np.abs(proj).reshape(16, 8), cmap='gray')
# plt.subplot(2, 3, 3)
# plt.imshow(np.abs(recon), cmap='gray')
# plt.subplot(2, 3, 4)
# plt.imshow(downsampled, cmap='gray')
# plt.subplot(2, 3, 5)
# plt.imshow(np.abs(downsampled_proj).reshape(16, 8), cmap='gray')
# plt.subplot(2, 3, 6)
# plt.imshow(np.abs(downsampled_recon), cmap='gray')
# plt.savefig("test.jpg")

# plt.imshow(np.abs(recon))
# plt.savefig("recon.jpg")

# a = np.arange(1, 17).reshape(4, 4)
# H_out, W_out = 2, 2
# k = np.ones((4, 1))
# i0 = np.repeat(np.arange(2), 2)
# i1 = 2 * np.repeat(np.arange(H_out), W_out)
# i = i0.reshape(-1, 1) + i1.reshape(1, -1)

# j0 = np.tile(np.arange(2), 2)
# j1 = 2 * np.tile(np.arange(W_out), H_out)
# j = j0.reshape(-1, 1) + j1.reshape(1, -1)
# print(a)
# print(i)
# print(j)
# print(a[i,j].sum(axis=0))
# print(H_reshape[:,i, j].shape)
# def svd(H_mn):
#     # H_adjoint_H = H_mn.conj().T.dot(H_mn)
#     # mu, u = np.linalg.eig(H_adjoint_H)
#     # v = H_mn.dot(u) / np.sqrt(mu)
#     H_adjoint = H_mn.conj().T
#     HH_adjoint = H.dot(H_adjoint)
#     mu, v = np.linalg.eig(HH_adjoint)
#     u = H_adjoint.dot(v) / np.sqrt(mu)
    
#     return mu, u, v

# def pseudo_inv(mu, u, v):
#     h_mp = np.zeros_like(v).T
#     for i in range(mu.shape[0]):
#         h_mp += u[:, i].reshape(-1, 1).dot(v[:, i].reshape(-1, 1).conj().T) / np.sqrt(mu[i])
#     # u_norm = u / np.sqrt(mu.reshape(1, -1))
#     # return u_norm.dot(v.conj().T)
#     return h_mp


# for num in range(100):

#     with open(f"../data/raw/{num}/orig.npy", "rb") as f:
#         sample = np.load(f)

#     obj = sample[0, ...]

#     # imsave("test.jpg", sample)

#     projection = (16, 8)
#     # H = generate_sys_matrix(64, projection)
#     H = generate_sys_matrix()

#     print(f"The rank of the operator: {np.linalg.matrix_rank(H)}")
#     proj = H.dot(obj.flatten())
#     H_MP = np.linalg.pinv(H)

#     print(H_MP.shape)

#     # print(H.shape)
#     # # print(f"The rank of the operator: {np.linalg.matrix_rank(H)}")
#     # # mu, u, v = svd(H)
#     # u, s, vh = np.linalg.svd(H, full_matrices=False)
#     # print(s)
#     # print(s.shape, u.shape, vh.shape)

#     # print(mu.shape, mu)

#     # print(u.shape, v.shape)
#     # H_MP = pseudo_inv(mu, u, v)
#     # print(H_MP.shape)
#     reconstruction = H_MP.dot(proj).reshape(64, 64)
#     plt.figure()
#     plt.subplot(1, 3, 1)
#     plt.imshow(obj, cmap='gray')
#     plt.axis('off')
#     plt.subplot(1, 3, 2)
#     plt.imshow(np.abs(proj).reshape(projection), cmap='gray')
#     plt.axis('off')
#     plt.subplot(1, 3, 3)
#     plt.imshow(np.abs(reconstruction), cmap='gray')
#     plt.axis('off')
#     # plt.colorbar(extend='both')
#     plt.savefig(f"test/{num}.jpg")
