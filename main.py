import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from data_loader.data_loader import CustomImageDataset
from setup import batch_size
import models.UNET as UNET

TrainingCustomImageDataset = CustomImageDataset()
print(len(TrainingCustomImageDataset))

# Display image and label.
train_dataloader = DataLoader(TrainingCustomImageDataset, batch_size=batch_size, shuffle=True)
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

train_features = train_features.permute(0, 3, 2, 1)

max_pool = torch.nn.MaxPool2d(kernel_size=(3, 3))
train_features = max_pool(train_features)

print(f"Max pooled features: {train_features.size()}")

features = [3, 64, 64]
model = UNET.UNET()
outs = model(train_features)
print(f'out shape: {outs.shape}')
