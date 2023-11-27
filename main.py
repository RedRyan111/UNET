import numpy as np
import torch
from PIL import Image

#im = Image.open(r"carvana-image-masking-challenge\train\train\0cdf5b5d0ce1_01.jpg")
#im.show()

mask = Image.open(r"carvana-image-masking-challenge\train_masks\train_masks\0cdf5b5d0ce1_01_mask.gif")
mask = np.array(mask.convert("L"), dtype=np.float32)
mask = torch.tensor(mask)

print(mask.shape)
