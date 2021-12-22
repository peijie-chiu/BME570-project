import os
from glob import glob
import numpy as np
from torch.utils.data import Dataset


class SegDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        with open(data_dir, "rb") as f:
            self.data = np.load(f)

        self.transform = transform
  
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        object, mask = np.float32(self.data[idx, :, :, 1]), self.data[idx, :, :, 2].astype(int)

        return object[np.newaxis, ...], mask[np.newaxis, ...]

