import os
from glob import glob
import numpy as np
from torch.utils import data
from torch.utils.data import Dataset, dataset


class SegDataset(Dataset):
    def __init__(self, img_dir, num_samples=500, transform=None):
        self.img_dir = img_dir
        self.num_samples = num_samples
        self.transform = transform
        self.Mean, self.Std = 6.3, 33.126

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        with open(f"{self.img_dir}/{str(idx)}/orig.npy", "rb") as f:
            data = np.load(f)

        object, mask = data[2, ...], data[3, ...] - 1

        return object[np.newaxis, ...], mask[np.newaxis, ...]


# dataset = SegDataset("data")
# # obj, mask = dataset[0]
# # print(obj.shape, mask.shape)

# psum = 0.
# psum_sq = 0.
# num_pixels = 500 * 64 * 64
# for i in range(500):
#     data = dataset[i][0]
#     psum += data.sum()
#     psum_sq += (data ** 2).sum()
    
# Mean = psum / num_pixels
# Var = (psum_sq / num_pixels) - (Mean ** 2)
# Std = np.sqrt(Var)

# print(Mean, Std)

# total_std = 0.
# for i in range(500):
#     data = dataset[i][0]
#     total_std += ((data-Mean) ** 2).sum()

# print(np.sqrt(total_std / num_pixels))