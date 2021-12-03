import os
import time
import numpy as np
import torch
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from utils import *
from dataloader import SegDataset


def generate_object(num_samples, s_range=(-1/2, 1/2), dim=(64, 64), out_dir="../data/new_raw", save_fig=True):
    print("start generating")
    try:
        os.mkdir(out_dir)
    except:
        pass 
    
    start = time.time()
    sig, sigma_s = multi_sig(3, dim=dim, s_range=s_range, num=num_samples)
    bgd = multi_backgrd(sigma_s, dim=dim, b_range=s_range)
    obj = sig + bgd
    end = time.time()
    print("Done !")
    print(f"Elapsed Time: {end-start:.2f}s")

    raw = np.stack([sig, bgd, obj])
    with open(f"{out_dir}/raw.npy", "wb") as f:
        np.save(f, raw)

    if save_fig:
        save_path = os.path.join(out_dir, "images")
        try:
            os.mkdir(save_path)
        except:
            pass 
        titles = ['signal', 'background', 'object', 'mask']
        for i in range(num_samples): 
            imgs = [sig[:,:,i], bgd[:,:,i], obj[:,:,i], sig[:,:,i]]
            save_figure(imgs, titles, f"{save_path}/{i}.jpg")

    return raw

############################## generate object ##############################
num_objects = 500
s_range = (-1/2, 1/2)
dim = (64, 64)
raw = generate_object(num_objects, s_range, dim)

############################## generate downsampled ##############################


############################## generate training/val/testing cache ##############################
# input_dir = "data/recon"
# save_dir = "cache/inputs/recon"

# try:
#     os.mkdir(save_dir)
# except:
#     pass

# dataset = SegDataset(input_dir)
# train_num = 300
# val_num = 100
# test_num = 100
# train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_num, val_num, test_num])
# torch.save(train_set, save_dir + "/train_set.pt")
# torch.save(val_set, save_dir + "/val_set.pt")
# torch.save(test_set, save_dir + "/test_set.pt")
