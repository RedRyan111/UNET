# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import unetConv2
from models.init_weights import init_weights
import torchvision.transforms.functional as TF
from down_samplers.UNET_MaxPool import DownSample


class DecoderBlock(nn.Module):
    def __init__(self, encoder_index, decoder_index, features, cat_channels, up_channels):
        super(DecoderBlock, self).__init__()

        inp_channels = features[encoder_index]
        if (encoder_index > decoder_index) and (encoder_index != len(features)-1):
            inp_channels = up_channels

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
        scaling_factor = 2 ** abs(decoder_level - encoder_level)
        if encoder_level == decoder_level:
            return ReturnInput()
        elif encoder_level < decoder_level:
            return DownSample(scaling_factor)
        else:
            return nn.UpsamplingBilinear2d(scale_factor=scaling_factor)

    def forward(self, x):
        return self.module(x)

class DecoderLevel(nn.Module):
    def __init__(self, features, decoder_level, CatChannels, UpChannels):
        super(DecoderLevel, self).__init__()
        self.module = nn.ModuleList()
        self.decoder_level = decoder_level
        self.features = features

        self.setup(features, decoder_level, CatChannels, UpChannels)

        self.conv4d_1 = nn.Conv2d(UpChannels, UpChannels, 3, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm2d(UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)

    def setup(self, features, decoder_level, CatChannels, UpChannels):
        for encoder_level in range(len(features)):
            self.module.append(DecoderBlock(encoder_index=encoder_level, decoder_index=decoder_level, features=features, cat_channels=CatChannels, up_channels=UpChannels))
    '''
    def forward(self, input_list):
        cur_out_list = []
        for i in range(len(self.features)):
            cur_inp = input_list[i]
            cur_model = self.module[i]
            cur_out = cur_model(cur_inp)

            if i !=0:
                cur_out = TF.resize(cur_out, cur_out_list[0].shape[2:])

            cur_out_list.append(cur_out)
    '''
    def forward(self, input_list):
        cur_out_list = []
        for i, (cur_inp, cur_model) in enumerate(zip(input_list, self.module)): #might be slow
            cur_out = cur_model(cur_inp)

            if i != 0:
                cur_out = TF.resize(cur_out, cur_out_list[0].shape[2:])

            cur_out_list.append(cur_out)

        return self.relu4d_1(self.bn4d_1(self.conv4d_1(torch.cat(cur_out_list, 1))))  # hd4->40*40*UpChannels


class Decoder(nn.Module):
    def __init__(self, filters, CatChannels, UpChannels):
        super(Decoder, self).__init__()
        self.decoder_nodes = None
        self.setup(filters, CatChannels, UpChannels)
        self.filter_length = len(filters)

    def setup(self, filters, CatChannels, UpChannels):
        self.decoder_nodes = []
        for decoder_level in range(len(filters) - 1):
            self.decoder_nodes.append(DecoderLevel(filters, decoder_level, CatChannels, UpChannels))

        self.decoder_nodes.reverse()
        self.decoder_nodes = nn.ModuleList(self.decoder_nodes)

    def forward(self, decoder_input):
        for decoder_level, decoder_node in enumerate(self.decoder_nodes):
            #change decoder input in reverse order
            decoder_input[self.filter_length - 2 - decoder_level] = decoder_node(decoder_input) #simpify this
        return decoder_input[0]


class UNet_3Plus(nn.Module):

    def __init__(self, in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(UNet_3Plus, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        # filters = [64, 128, 256, 512, 1024]
        #filters = [16, 32, 64, 128, 256]
        filters = [4, 8, 16, 32, 64]
        self.filters = filters
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        ## -------------Encoder--------------
        self.encoder = Encoder(filters)

        ## -------------Decoder--------------
        self.decoder = Decoder(filters, self.CatChannels, self.UpChannels)

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
        decoder_input = self.encoder(inputs)

        ## -------------Decoder-------------
        hd1 = self.decoder(decoder_input)

        d1 = self.outconv1(hd1)  # d1->320*320*n_classes
        return torch.sigmoid(d1)


class Encoder(nn.Module):
    def __init__(self, filters):
        super(Encoder, self).__init__()

        self.module = nn.ModuleList([])
        self.setup(filters)

    def setup(self, filters):
        for index, feature in enumerate(filters):
            is_first = index == 0
            self.module.append(EncoderBlock(filters, index, is_first))

    def forward(self, x):
        encoder_outputs = [x]
        for index, module in enumerate(self.module):
            cur_output = module(encoder_outputs[index])
            encoder_outputs.append(cur_output)

        return encoder_outputs[1:]


class EncoderBlock(nn.Module):
    def __init__(self, filters, encoder_level, is_first=False,  isBatchNorm=True):
        super(EncoderBlock, self).__init__()

        self.module = self.setup(filters, encoder_level, is_first, isBatchNorm)

    def setup(self, filters, encoder_level, is_first, isBatchNorm):
        if is_first:
            return unetConv2(3, filters[encoder_level], isBatchNorm)
        else:
            return nn.Sequential(
                nn.MaxPool2d(kernel_size=2),
                unetConv2(filters[encoder_level-1], filters[encoder_level], isBatchNorm)
            )

    def forward(self, x):
        return self.module(x)
