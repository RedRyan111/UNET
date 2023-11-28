import torch.nn as nn
import torch


class UNET(nn.Module):
    def __init__(self):
        super(UNET, self).__init__()

    def forward(self, x):
        return x


class DoubleConv(nn.Module):
    def __init__(self, features):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(features[0], features[1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(features[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[1], features[2], 3, 1, 1, bias=False),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super(DownSample, self).__init__()
        # downsampling by 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        )

    def forward(self, x):
        return self.conv(x)
