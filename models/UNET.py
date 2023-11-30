import numpy as np
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional
import torch.nn.functional as F


class UNET(nn.Module):
    def __init__(self):
        super(UNET, self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up_sample = UpSample(scaling_factor=2)

        self.down_sample = DownSample()

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
        print(f'x pre shape: {x.shape}')
        x = pad_tensor_shapes_if_odd(x)
        print(f'x post shape: {x.shape}')
        print(f'')

        y1 = self.down_sample(self.double_1(x))
        print(f'y1 pre shape: {y1.shape}')
        y1 = pad_tensor_shapes_if_odd(y1)
        print(f'y1 post shape: {y1.shape}')
        print(f'')

        y2 = self.down_sample(self.double_2(y1))
        print(f'y2: {y2.shape}')
        y3 = self.down_sample(self.double_3(y2))
        print(f'y3: {y3.shape}')
        y4 = self.down_sample(self.double_4(y3))
        print(f'y4: {y4.shape}')
        y5 = self.down_sample(self.double_5(y4))
        print(f'y5: {y5.shape}')

        #y5_up = torch.concatenate((y4, y5), dim=-1)
        y6 = self.up_sample(self.double_6(y5))

        print(f'y6 pre crop: {y6.shape}')
        y6 = torchvision.transforms.functional.center_crop(img=y6, output_size=[y4.shape[2], y4.shape[3]]) #permute?
        print(f'y6 post crop: {y6.shape}')

        y6_up = torch.concatenate((y3, y6), dim=-1)
        y7 = self.up_sample(self.double_7(y2, y6_up))

        y7_up = torch.concatenate((y2, y7), dim=-1)
        y8 = self.up_sample(self.double_8(y2, y7_up))

        y8_up = torch.concatenate((y1, y8), dim=-1)
        y9 = self.up_sample(self.double_9(y2, y8_up))

        return y9


def pad_tensor_shapes_if_odd(inp_tensor):
    tensor_shape = [0 for i in range(4)]
    tensor_shape[2] = inp_tensor.shape[2] % 2
    tensor_shape[3] = inp_tensor.shape[3] % 2
    return F.pad(inp_tensor, tensor_shape, "constant", 0)

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
    def __init__(self):
        super(DownSample, self).__init__()

        self.down_sample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.down_sample(x)

class UpSample(nn.Module):
    def __init__(self, scaling_factor):
        super(UpSample, self).__init__()

        self.up_sample = nn.UpsamplingBilinear2d(scale_factor=scaling_factor)

    def forward(self, x):
        return self.up_sample(x)
