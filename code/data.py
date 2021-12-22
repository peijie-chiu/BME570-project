import os
import time
import numpy as np
import torch
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from utils import *
from imaging import *
from dataloader import SegDataset

np.random.seed(42)


def generate_object(num_samples, s_range=(-1/2, 1/2), dim=(64, 64), out_dir="../data", save_fig=True):
    print("start generating")
    start = time.time()
    sig, sigma_s = multi_sig(3, dim=dim, s_range=s_range, num=num_samples)
    bgd = multi_backgrd(sigma_s, dim=dim, b_range=s_range)
    obj = sig + bgd
    end = time.time()
    print("Done !")
    print(f"Elapsed Time: {end-start:.2f}s")

    raw = np.stack([sig, bgd, obj])

    if out_dir:
        save_path = os.path.join(out_dir, "raw")
        try:
            os.mkdir(save_path)
        except:
            pass 

        with open(f"{save_path}/data.npy", "wb") as f:
            np.save(f, raw)

        with open(f"{save_path}/sigma.npy", "wb") as f:
            np.save(f, sigma_s)

        if save_fig:
            titles = ['signal', 'background', 'object', 'mask']
            for i in range(num_samples): 
                imgs = [sig[:,:,i], bgd[:,:,i], obj[:,:,i], sig[:,:,i]]
                save_figure(imgs, titles, f"{save_path}/{i}.jpg")

    return raw


def generate_recon(obj, mask, downsample_factor=2, out_dir="../data", save_fig=True):
    print("start generating")
    start = time.time()
    H = generate_sys_matrix(downsample_factor=downsample_factor)
    downsampled_obj, downsampled_mask = downsample(obj, mask, downsample_factor)
    proj, recon = reconstruction(downsampled_obj, H)
    end = time.time()
    print("Done !")
    print(f"Elapsed Time: {end-start:.2f}s")

    raw = np.stack([downsampled_obj, recon, downsampled_mask])
    if out_dir:
        save_path = os.path.join(out_dir, f"{64//downsample_factor}x{64//downsample_factor}")
        try:
            os.mkdir(save_path)
        except:
            pass
        
        with open(f"{save_path}/data.npy", "wb") as f:
            np.save(f, raw)
            np.save(f, proj)

        if save_fig:
            image_path = os.path.join(save_path, "images") 
            try:
                os.mkdir(image_path)
            except:
                pass
            titles = ['obj', 'projection', 'reconstruction', 'mask']
            for i in range(obj.shape[2]): 
                imgs = [downsampled_obj[:,:,i], proj[:,:,i], recon[:,:,i], downsampled_mask[:,:,i]]
                save_figure(imgs, titles, f"{image_path}/{i}.jpg")


############################## generate object ##############################
num_objects = 500
s_range = (-1/2, 1/2)
dim = (64, 64)
raw = generate_object(num_objects, s_range, dim, save_fig=True)
obj = raw[2, ...]
mask = raw[0, ...]
mask = np.where(mask > 0, mask - 2, mask)

downsample_factors = [1, 2, 4, 8]
for factor in downsample_factors:
    generate_recon(obj, mask, downsample_factor=factor, save_fig=True)


############################## generate training/val/testing cache ##############################
# dim = 64
# tr_samples = 300
# val_samples = 100
# test_samples = 100
# downsample_factor = [1, 2, 4, 8]
# input_dir = "../data"

# for factor in downsample_factor:
#     input_path = f"{input_dir}/{dim//factor}x{dim//factor}"
#     with open(f"{input_path}/data.npy", "rb") as f:
#         data = np.load(f)

#     dataset = data.transpose(3, 1, 2, 0)
#     train_set = dataset[tr_samples:, ...]
#     val_set = dataset[tr_samples: (tr_samples + val_samples), ...]
#     test_set = dataset[(tr_samples + val_samples):, ...]

#     print(train_set.shape, val_set.shape, test_set.shape)

#     with open(f"{input_path}/train_set.npy", "wb") as f:
#         np.save(f, train_set)

#     with open(f"{input_path}/val_set.npy", "wb") as f:
#         np.save(f, val_set)

#     with open(f"{input_path}/test_set.npy", "wb") as f:
#         np.save(f, test_set)
    
# input_path = "data/32x32"
# with open(f"{input_path}/train_set.npy", "wb") as f:
#     train_set = np.load(f)

# with open(f"{input_path}/val_set.npy", "wb") as f:
#     val_set = np.save(f)

# with open(f"{input_path}/test_set.npy", "wb") as f:
#     test_set = np.save(f)

# print(train_set.shape)