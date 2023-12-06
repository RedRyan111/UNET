from torch.utils.data import DataLoader
from data_loader.data_loader import CustomImageDataset
from setup import batch_size
import models.UNET as UNET
import torch.nn as nn
import torch.optim as op

TrainingCustomImageDataset = CustomImageDataset()
print(len(TrainingCustomImageDataset))

train_dataloader = DataLoader(TrainingCustomImageDataset, batch_size=batch_size, shuffle=True)

features = [12, 24, 48]
num_inp_channels = 3
num_labels = 1
model = UNET.UNET(features, num_inp_channels, num_labels)

mse_loss = nn.MSELoss()
optimizer = op.Adam(model.parameters(), lr=.001)

epochs = 100

for epoch in range(epochs):
    train_features, train_labels = next(iter(train_dataloader))
    train_features = train_features.permute(0, 3, 2, 1)

    outs = model(train_features)

    outs = outs.permute(0, 3, 2, 1).squeeze()

    loss = mse_loss(train_labels, outs)
    loss.backward()
    optimizer.step()
    print(f'epoch: {epoch} loss: {loss}')

print('DONE!')
