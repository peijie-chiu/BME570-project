import torch
from torch.utils.data import DataLoader

from dataloader import SegDataset


input_dir = "data"

dataset = SegDataset(input_dir)
train_num = 300
val_num = 100
test_num = 100