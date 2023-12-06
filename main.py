from torch.utils.data import DataLoader
from data_loader.data_loader import CustomImageDataset
import models.UNET as UNET
import torch.nn as nn
import torch.optim as op
import torch

TrainingCustomImageDataset = CustomImageDataset()
print(len(TrainingCustomImageDataset))
batch_size = 20

train_dataloader = DataLoader(TrainingCustomImageDataset, batch_size=batch_size, shuffle=True)

features = [12, 24, 48, 96]
num_inp_channels = 3
num_labels = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')

model = UNET.UNET(features, num_inp_channels, num_labels).to(device)

mse_loss = nn.MSELoss() #change to BCE
optimizer = op.Adam(model.parameters(), lr=.001)
max_pool = nn.MaxPool2d(kernel_size=(3, 3))
epochs = 6000

for epoch in range(epochs):
    train_features, train_labels = next(iter(train_dataloader))
    train_labels = train_labels.to(device)
    train_features = train_features.to(device).permute(0, 3, 2, 1)

    train_features = max_pool(train_features)
    train_labels = max_pool(train_labels)
    outs = model(train_features)

    outs = outs.permute(0, 3, 2, 1).squeeze()

    loss = mse_loss(train_labels, outs)
    loss.backward()
    optimizer.step()
    print(f'epoch: {epoch} loss: {loss}')

print('DONE!')
