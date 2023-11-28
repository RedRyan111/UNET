import numpy as np
import torch.nn as nn
import torch


class UNET(nn.Module):
    def __init__(self):
        super(UNET, self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        feature_1 = np.array([3, 32, 32])
        self.double_1 = DoubleConv(feature_1)

        feature_2 = np.array([64, 128, 128]) // 2
        self.double_2 = DoubleConv(feature_2)

        feature_3 = np.array([128, 256, 256]) // 2
        self.double_3 = DoubleConv(feature_3)

        feature_4 = np.array([256, 512, 512]) // 2
        self.double_4 = DoubleConv(feature_4)

        feature_5 = np.array([512, 1024, 1024]) // 2
        self.double_5 = DoubleConv(feature_5)

        double_6_features = np.flip(feature_4)
        double_6_features[0] = double_6_features[0] * 2

        double_7_features = np.flip(feature_3)
        double_7_features[0] = double_7_features[0] * 2

        double_8_features = np.flip(feature_2)
        double_8_features[0] = double_6_features[0] * 2

        double_9_features = np.flip(feature_1)
        double_9_features[0] = double_9_features[0] * 2

        self.double_6 = DoubleConv(np.flip(feature_5))
        self.double_7 = DoubleConv(np.flip(feature_4))
        self.double_8 = DoubleConv(np.flip(feature_3))
        self.double_9 = DoubleConv(np.flip(feature_2))


    def forward(self, x):
        y1 = self.max_pool(self.double_1(x))
        y2 = self.max_pool(self.double_2(y1))
        y3 = self.max_pool(self.double_3(y2))
        y4 = self.max_pool(self.double_4(y3))
        y5 = self.max_pool(self.double_5(y4))

        y1 = self.max_pool(self.up_1(x))
        y2 = self.max_pool(self.up_2(y1))
        y3 = self.max_pool(self.up_3(y2))
        y4 = self.max_pool(self.up_4(y3))
        y5 = self.max_pool(self.up_5(y4))

        return y5


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

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super(UpSample, self).__init__()
        # downsampling by 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        )

    def forward(self, x):
        return self.conv(x)