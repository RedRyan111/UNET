from typing import List

import numpy as np
import torch.nn as nn
import torch
import torchvision.transforms.functional
import torch.nn.functional as F
import torchvision.transforms.functional as TF


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
            print(f'encoder features: {double_conv_features}')
            double_conv = DoubleConv(double_conv_features)
            self.module.append(double_conv)

    def forward(self, x):
        out_list = []
        cur_out = x
        for module in self.module:
            cur_out = module(cur_out)
            h = cur_out.shape[2] + cur_out.shape[2] % 2
            w = cur_out.shape[3] + cur_out.shape[3] % 2
            padded_out = TF.resize(cur_out, size=[h, w])

            #padded_out = pad_tensor_shapes_if_odd(cur_out)
            #print(f'prev out: {cur_out.shape} padded out: {padded_out.shape}')
            out_list.append(padded_out)

            cur_out = self.down_sampler(padded_out)
        return out_list


class Decoder(nn.Module):
    def __init__(self, features):
        super(Decoder, self).__init__()
        self.module = nn.ModuleList([])
        self.up_sampler = nn.ModuleList([])

        features.reverse()

        self.setup_double_conv(features)
        self.setup_up_sampler(features)

    def setup_double_conv(self, features):
        for i in range(len(features)-1):
            double_conv_features = [features[i], features[i+1], features[i+1]]
            print(f'decoder features: {double_conv_features}')
            double_conv = DoubleConv(double_conv_features)
            self.module.append(double_conv)

    def setup_up_sampler(self, features):
        for i in range(len(features)-2):
            cur_up_sampler = UpSample(2, features[i], features[i+1])
            self.up_sampler.append(cur_up_sampler)

    def forward(self, out_list: List):
        out_list.reverse()
        up_sample_model = self.up_sampler[0]
        cur_up_sample = up_sample_model(out_list[0])
        for i in range(len(out_list)-1):
            next_encoder_output = out_list[i+1]
            #cur_up_sample = crop_tensor_to_tensor_h_and_w(cur_up_sample, next_encoder_output)
            cur_up_sample = TF.resize(cur_up_sample, next_encoder_output.shape[2:])

            double_conv_inp = torch.concatenate((cur_up_sample, next_encoder_output), dim=1)

            double_conv = self.module[i]
            cur_double_conv = double_conv(double_conv_inp)

            if i == len(out_list)-2:
                double_conv = self.module[i+1]
                cur_double_conv = double_conv(cur_double_conv)
                cur_double_conv = crop_tensor_to_tensor_h_and_w(cur_double_conv, next_encoder_output)
                return cur_double_conv

            up_sample_model = self.up_sampler[i+1]
            cur_up_sample = up_sample_model(cur_double_conv) #this is causing problems


class UNET(nn.Module):
    def __init__(self):
        super(UNET, self).__init__()

        self.features = [3, 12, 24, 48]#[3, 64, 128, 256, 512, 1024]

        self.down_sample = DownSample(scaling_factor=2)

        self.encoder = Encoder(self.features, self.down_sample)
        self.decoder = Decoder(self.features)

    def forward(self, x):
        out_list = self.encoder(x)
        y = self.decoder(out_list)
        y = crop_tensor_to_tensor_h_and_w(y, x)
        return y


def crop_tensor_to_tensor_h_and_w(x, y):
    return torchvision.transforms.functional.center_crop(img=x, output_size=[y.shape[2], y.shape[3]])


def pad_tensor_shapes_if_odd(inp_tensor):
    tensor_shape = [0 for _ in range(4)]
    tensor_shape[2] = inp_tensor.shape[2] % 2
    tensor_shape[0] = inp_tensor.shape[3] % 2
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
        self.conv = nn.Conv2d(inp_features, out_features, kernel_size=3, padding=1)

    def forward(self, x):
        up = self.up_sample(x)
        conv = self.conv(up)
        print(f'inp: {x.shape} up: {up.shape} conv: {conv.shape}')
        return conv
