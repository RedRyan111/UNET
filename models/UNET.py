from typing import List

import numpy as np
import torch.nn as nn
import torch
import torchvision.transforms.functional
import torch.nn.functional as F


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


class Encoder(nn.Module):
    def __init__(self, features, down_sampler):
        super(Encoder, self).__init__()
        self.down_sampler = down_sampler
        self.module = nn.ModuleList([])
        self.setup_module(features)

    def setup_module(self, features):
        feature_length = len(features)
        for i in range(feature_length-1):
            double_conv_features = [features[i], features[i+1], features[i+1]]
            print(f'encoder: {double_conv_features}')
            double_conv = DoubleConv(double_conv_features)
            self.module.append(double_conv)

    def forward(self, x):
        out_list = []
        cur_out = x
        for module in self.module:
            cur_out = module(cur_out)
            cur_out = pad_tensor_shapes_if_odd(cur_out)
            out_list.append(cur_out)

            cur_out = self.down_sampler(cur_out)
        return out_list


class Decoder(nn.Module):
    def __init__(self, features):
        super(Decoder, self).__init__()
        self.module = nn.ModuleList([])
        self.up_sampler = nn.ModuleList([])

        self.setup_decoder(features)

    def setup_decoder(self, features):
        feature_length = len(features)
        for i in range(feature_length-2, 0, -1):
            double_conv_features = [features[i]*2, features[i], features[i]]
            print(f'decoder feature: {double_conv_features}')
            double_conv = DoubleConv(double_conv_features)
            self.module.append(double_conv)

            cur_upsampler = UpSample(2, features[i]*2, features[i])
            self.up_sampler.append(cur_upsampler)

        double_conv_features = [features[1], features[0], features[0]]
        print(f'decoder feature: {double_conv_features}')
        double_conv = DoubleConv(double_conv_features)
        self.module.append(double_conv)

        cur_upsampler = UpSample(2, features[1], features[0])
        self.up_sampler.append(cur_upsampler)

    def forward(self, out_list: List):
        out_list.reverse()

        for i in range(len(out_list)-1):
            up_sample_model = self.up_sampler[i]
            encoder_output = out_list[i]
            cur_upsample = up_sample_model(encoder_output)

            next_encoder_output = out_list[i+1]
            next_encoder_output = crop_tensor_to_tensor_h_and_w(next_encoder_output, cur_upsample)

            print(f'upsample: {cur_upsample.shape} next encoder: {next_encoder_output.shape}')

            double_conv_inp = torch.concatenate((cur_upsample, next_encoder_output), dim=1)

            double_conv = self.module[i]

            return double_conv(double_conv_inp)


class UNET(nn.Module):
    def __init__(self):
        super(UNET, self).__init__()

        self.features = [3, 32, 64, 128]

        #self.up_sample = UpSample(scaling_factor=2)
        self.down_sample = DownSample(scaling_factor=2)

        self.encoder = Encoder(self.features, self.down_sample)
        self.decoder = Decoder(self.features)#, self.up_sample)

    def forward(self, x):
        out_list = self.encoder(x)
        [print(i.shape) for i in out_list]
        y = self.decoder(out_list)
        return y


def crop_tensor_to_tensor_h_and_w(x, y):
    return torchvision.transforms.functional.center_crop(img=x, output_size=[y.shape[2], y.shape[3]])

'''
def resize(x, skip_connection):
    if x.shape != skip_connection.shape:
        x = TF.resize(x, size=skip_connection.shape[2:])
    return x
'''


def pad_tensor_shapes_if_odd(inp_tensor):
    tensor_shape = [0 for i in range(4)]
    tensor_shape[2] = inp_tensor.shape[2] % 2
    tensor_shape[3] = inp_tensor.shape[3] % 2
    return F.pad(inp_tensor, tensor_shape, "constant", 0)


class DownSample(nn.Module):
    def __init__(self, scaling_factor):
        super(DownSample, self).__init__()

        self.down_sample = nn.MaxPool2d(kernel_size=scaling_factor, stride=scaling_factor)

    def forward(self, x):
        return self.down_sample(x)


class UpSample(nn.Module):
    def __init__(self, scaling_factor, inp_features, out_features):
        super(UpSample, self).__init__()

        self.up_sample = nn.UpsamplingBilinear2d(scale_factor=scaling_factor)
        self.conv = nn.Conv2d(inp_features, out_features, kernel_size=3)

    def forward(self, x):
        up = self.up_sample(x)
        conv = self.conv(up)
        return conv
