import numpy as np


flip_angles = (np.pi / 8) * np.arange(0, 8)
m = (np.arange(1, 17, 1) - 8).reshape(-1, 1)

km_x = m * np.cos(flip_angles)
km_y = m * np.sin(flip_angles)
km = np.stack([km_x, km_y])
print(km[:, 0, 0])
