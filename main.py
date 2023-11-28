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

features = [3, 64, 64]
model = UNET.DoubleConv(features)
outs = model(train_features)
print(f'out shape: {outs.shape}')
