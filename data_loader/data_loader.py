import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os

# im = Image.open(r"carvana-image-masking-challenge\train\train\0cdf5b5d0ce1_01.jpg")
# im.show()

print(f"inside data loader: {os.getcwd()}")

mask = Image.open(r"carvana-image-masking-challenge/train_masks/train_masks/0cdf5b5d0ce1_01_mask.gif")
mask = np.array(mask.convert("L"), dtype=np.float32)
mask = torch.tensor(mask)

print(mask.shape)


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.train_images_dir_name = "carvana-image-masking-challenge/train/train/"
        self.train_masks_dir_name = "carvana-image-masking-challenge/train_masks/train_masks/"

        self.train_images_file_name_list = os.listdir(train_images_dir_name)
        self.train_masks_file_name_list = os.listdir(train_masks_dir_name)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.train_images_dir_name)

    def get_image(self, idx):
        file_name = self.train_images_file_name_list[idx]
        image = Image.open(file_name)
        image_tensor = torch.tensor(image)
        return image_tensor

    def get_mask(self, idx):
        file_name = self.train_masks_file_name_list[idx]
        mask = Image.open(file_name)
        mask = np.array(mask.convert("L"), dtype=np.float32)
        mask = torch.tensor(mask)
        return mask

    def __getitem__(self, idx):
        image = self.get_image(idx)
        mask = self.get_mask(idx)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            mask = self.target_transform(mask)
        return image, mask


train_images_dir_name = "carvana-image-masking-challenge/train/train/"
train_masks_dir_name = "carvana-image-masking-challenge/train_masks/train_masks/"

train_images_file_name_list = os.listdir(train_images_dir_name)
train_masks_file_name_list = os.listdir(train_masks_dir_name)
print(len(train_images_file_name_list))
print(len(train_masks_file_name_list))
