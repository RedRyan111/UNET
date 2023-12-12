# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
# from layers import unetConv2
# from init_weights import init_weights
from models1.layers import unetConv2
from models1.init_weights import init_weights
import torchvision.transforms.functional as TF
from down_samplers.UNET_MaxPool import DownSample
#from up_samplers.UNET_Bilinear import UpSample


class EncoderBlock(nn.Module):
    def __init__(self, encoder_index, decoder_index, features, cat_channels, up_channels):
        super(EncoderBlock, self).__init__()

        inp_channels = features[encoder_index]
        if (encoder_index > decoder_index) and (encoder_index != len(features)-1):
            inp_channels = up_channels

        #print(f'encoder_index: {encoder_index} decoder_index: {decoder_index} inp channels: {inp_channels} cat channels: {cat_channels}')

        self.module = nn.Sequential(
            UpOrDownSample(encoder_index, decoder_index),
            nn.Conv2d(inp_channels, cat_channels, 3, padding=1),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.module(x)


class ReturnInput(nn.Module):
    def __init__(self):
        super(ReturnInput, self).__init__()

    def forward(self, x):
        return x


class UpOrDownSample(nn.Module):
    def __init__(self, encoder_index, decoder_index):
        super(UpOrDownSample, self).__init__()
        self.module = self.get_scaling_module(encoder_index, decoder_index)

    def get_scaling_module(self, encoder_level, decoder_level):
        #scaling_factor = 2 ** torch.abs(decoder_level - encoder_level)
        if encoder_level == decoder_level:
            #print(f'No Scaling')
            return ReturnInput()
        elif encoder_level < decoder_level:
            scaling_factor = 2 ** (decoder_level - encoder_level)
            #print(f'downscaling by: {scaling_factor}')
            return DownSample(scaling_factor)
        else:
            scaling_factor = 2 ** (encoder_level - decoder_level)
            #print(f'scale factor: {scaling_factor}')
            return nn.UpsamplingBilinear2d(scale_factor=scaling_factor)

    def forward(self, x):
        return self.module(x)


class UNet_3Plus(nn.Module):

    def __init__(self, in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(UNet_3Plus, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        # filters = [64, 128, 256, 512, 1024]
        # filters = [8, 16, 32, 64, 128, 256]
        filters = [2, 4, 8, 16, 32]

        ## -------------Encoder--------------
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 4d'''
        self.block_0_to_3 = EncoderBlock(encoder_index=0, decoder_index=3, features=filters, cat_channels=self.CatChannels, up_channels=self.UpChannels)
        self.block_1_to_3 = EncoderBlock(encoder_index=1, decoder_index=3, features=filters, cat_channels=self.CatChannels, up_channels=self.UpChannels)
        self.block_2_to_3 = EncoderBlock(encoder_index=2, decoder_index=3, features=filters, cat_channels=self.CatChannels, up_channels=self.UpChannels)
        self.block_3_to_3 = EncoderBlock(encoder_index=3, decoder_index=3, features=filters, cat_channels=self.CatChannels, up_channels=self.UpChannels)
        self.block_4_to_3 = EncoderBlock(encoder_index=4, decoder_index=3, features=filters, cat_channels=self.CatChannels, up_channels=self.UpChannels)

        #Decoder block
        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)

        '''stage 3d'''
        self.block_0_to_2 = EncoderBlock(encoder_index=0, decoder_index=2, features=filters, cat_channels=self.CatChannels, up_channels=self.UpChannels)
        self.block_1_to_2 = EncoderBlock(encoder_index=1, decoder_index=2, features=filters, cat_channels=self.CatChannels, up_channels=self.UpChannels)
        self.block_2_to_2 = EncoderBlock(encoder_index=2, decoder_index=2, features=filters, cat_channels=self.CatChannels, up_channels=self.UpChannels)
        self.block_3_to_2 = EncoderBlock(encoder_index=3, decoder_index=2, features=filters, cat_channels=self.CatChannels, up_channels=self.UpChannels)
        self.block_4_to_2 = EncoderBlock(encoder_index=4, decoder_index=2, features=filters, cat_channels=self.CatChannels, up_channels=self.UpChannels)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        '''stage 2d '''
        self.block_0_to_1 = EncoderBlock(encoder_index=0, decoder_index=1, features=filters, cat_channels=self.CatChannels, up_channels=self.UpChannels)
        self.block_1_to_1 = EncoderBlock(encoder_index=1, decoder_index=1, features=filters, cat_channels=self.CatChannels, up_channels=self.UpChannels)
        self.block_2_to_1 = EncoderBlock(encoder_index=2, decoder_index=1, features=filters, cat_channels=self.CatChannels, up_channels=self.UpChannels)
        self.block_3_to_1 = EncoderBlock(encoder_index=3, decoder_index=1, features=filters, cat_channels=self.CatChannels, up_channels=self.UpChannels)
        self.block_4_to_1 = EncoderBlock(encoder_index=4, decoder_index=1, features=filters, cat_channels=self.CatChannels, up_channels=self.UpChannels)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        '''stage 1d'''
        self.block_0_to_0 = EncoderBlock(encoder_index=0, decoder_index=0, features=filters, cat_channels=self.CatChannels, up_channels=self.UpChannels)
        self.block_1_to_0 = EncoderBlock(encoder_index=1, decoder_index=0, features=filters, cat_channels=self.CatChannels, up_channels=self.UpChannels)
        self.block_2_to_0 = EncoderBlock(encoder_index=2, decoder_index=0, features=filters, cat_channels=self.CatChannels, up_channels=self.UpChannels)
        self.block_3_to_0 = EncoderBlock(encoder_index=3, decoder_index=0, features=filters, cat_channels=self.CatChannels, up_channels=self.UpChannels)
        self.block_4_to_0 = EncoderBlock(encoder_index=4, decoder_index=0, features=filters, cat_channels=self.CatChannels, up_channels=self.UpChannels)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        # output
        self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        ## -------------Encoder-------------
        h1 = self.conv1(inputs)  # h1->320*320*64

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->160*160*128

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->80*80*256

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->40*40*512

        h5 = self.maxpool4(h4)
        hd5 = self.conv5(h5)  # h5->20*20*1024

        ## -------------Decoder-------------

        h1_PT_hd4 = self.block_0_to_3(h1)
        h2_PT_hd4 = self.block_1_to_3(h2)
        h3_PT_hd4 = self.block_2_to_3(h3)
        h4_Cat_hd4 = self.block_3_to_3(h4)
        hd5_UT_hd4 = self.block_4_to_3(hd5)

        h2_PT_hd4 = TF.resize(h2_PT_hd4, h1_PT_hd4.shape[2:])
        h3_PT_hd4 = TF.resize(h3_PT_hd4, h1_PT_hd4.shape[2:])
        h4_Cat_hd4 = TF.resize(h4_Cat_hd4, h1_PT_hd4.shape[2:])
        hd5_UT_hd4 = TF.resize(hd5_UT_hd4, h1_PT_hd4.shape[2:])

        # print(f'{h1_PT_hd4.shape} {h2_PT_hd4.shape} {h3_PT_hd4.shape} {h4_Cat_hd4.shape} {hd5_UT_hd4.shape}')

        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1))))  # hd4->40*40*UpChannels

        #print(f'hd4: {hd4.shape}')

        h1_PT_hd3 = self.block_0_to_2(h1)
        h2_PT_hd3 = self.block_1_to_2(h2)
        h3_Cat_hd3 = self.block_2_to_2(h3)
        hd4_UT_hd3 = self.block_3_to_2(hd4)
        hd5_UT_hd3 = self.block_4_to_2(hd5)

        h2_PT_hd3 = TF.resize(h2_PT_hd3, h1_PT_hd3.shape[2:])
        h3_Cat_hd3 = TF.resize(h3_Cat_hd3, h1_PT_hd3.shape[2:])
        hd4_UT_hd3 = TF.resize(hd4_UT_hd3, h1_PT_hd3.shape[2:])
        hd5_UT_hd3 = TF.resize(hd5_UT_hd3, h1_PT_hd3.shape[2:])

        # print(f'{h1_PT_hd3.shape} {h2_PT_hd3.shape} {h3_Cat_hd3.shape} {hd4_UT_hd3.shape} {hd5_UT_hd3.shape}')

        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1))))  # hd3->80*80*UpChannels

        h1_PT_hd2 = self.block_0_to_1(h1)
        h2_Cat_hd2 = self.block_1_to_1(h2)
        hd3_UT_hd2 = self.block_2_to_1(hd3)
        hd4_UT_hd2 = self.block_3_to_1(hd4)
        hd5_UT_hd2 = self.block_4_to_1(hd5)

        h2_Cat_hd2 = TF.resize(h2_Cat_hd2, h1_PT_hd2.shape[2:])
        hd3_UT_hd2 = TF.resize(hd3_UT_hd2, h1_PT_hd2.shape[2:])
        hd4_UT_hd2 = TF.resize(hd4_UT_hd2, h1_PT_hd2.shape[2:])
        hd5_UT_hd2 = TF.resize(hd5_UT_hd2, h1_PT_hd2.shape[2:])

        # print(f'{h2_Cat_hd2.shape} {h2_Cat_hd2.shape} {hd3_UT_hd2.shape} {hd4_UT_hd2.shape} {hd5_UT_hd2.shape}')

        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1))))  # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.block_0_to_0(h1)
        hd2_UT_hd1 = self.block_1_to_0(hd2)
        hd3_UT_hd1 = self.block_2_to_0(hd3)
        hd4_UT_hd1 = self.block_3_to_0(hd4)
        hd5_UT_hd1 = self.block_4_to_0(hd5)

        hd2_UT_hd1 = TF.resize(hd2_UT_hd1, h1_Cat_hd1.shape[2:])
        hd3_UT_hd1 = TF.resize(hd3_UT_hd1, h1_Cat_hd1.shape[2:])
        hd4_UT_hd1 = TF.resize(hd4_UT_hd1, h1_Cat_hd1.shape[2:])
        hd5_UT_hd1 = TF.resize(hd5_UT_hd1, h1_Cat_hd1.shape[2:])

        # print(f'{h1_Cat_hd1.shape} {hd2_UT_hd1.shape} {hd3_UT_hd1.shape} {hd4_UT_hd1.shape} {hd5_UT_hd1.shape}')

        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1))))  # hd1->320*320*UpChannels

        d1 = self.outconv1(hd1)  # d1->320*320*n_classes
        return torch.sigmoid(d1)
