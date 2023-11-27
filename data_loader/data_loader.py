import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import os


class CustomImageDataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.train_images_dir_name = "carvana-image-masking-challenge/train/train/"
        self.train_masks_dir_name = "carvana-image-masking-challenge/train_masks/train_masks/"

        self.train_images_file_name_list = os.listdir(self.train_images_dir_name)
        self.train_masks_file_name_list = os.listdir(self.train_masks_dir_name)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.train_images_file_name_list)

    def get_image(self, idx):
        file_name = self.train_images_dir_name + self.train_images_file_name_list[idx]
        image = Image.open(file_name)
        image = np.array(image.convert("L"), dtype=np.float32)
        image_tensor = torch.tensor(image)
        return image_tensor

    def get_mask(self, idx):
        file_name = self.train_masks_dir_name + self.train_masks_file_name_list[idx]
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


TrainingCustomImageDataset = CustomImageDataset()
print(len(TrainingCustomImageDataset))

# Display image and label.
train_dataloader = DataLoader(TrainingCustomImageDataset, batch_size=64, shuffle=True)
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

