#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/16 15:36
# @Author  : Shenhan Qian
# @Site    : 
# @File    : lanenet.py
# @Software: PyCharm


import torch
import torch.nn as nn
from model import decoder_fcn
from model.resnet import resnet18, resnet34
from model.enet import ENet_encoder, ENet_decoder, ENet
from model.icnet import ICNet_1E2D, ICNet

class LaneNet_FCN_Res_1E1D(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = resnet18(pretrained=True)
        # self.encoder = resnet34(pretrained=True)
        self.decoder = decoder_fcn.Decoder_LaneNet_TConv()
        # self.decoder = decoder_fcn.Decoder_LaneNet_Interplt()

    def forward(self, input):
        x = self.encoder.forward(input)

        input_tensor_list = [self.encoder.c1, self.encoder.c2, self.encoder.c3, self.encoder.c4, x]
        embedding, logit = self.decoder.forward(input_tensor_list)

        return embedding, logit


class LaneNet_FCN_Res_1E2D(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = resnet18(pretrained=True)
        # self.encoder = resnet34(pretrained=True)
        self.decoder_embed = decoder_fcn.Decoder_LaneNet_TConv_Embed()
        self.decoder_logit = decoder_fcn.Decoder_LaneNet_TConv_Logit()
        # self.decoder = decoder_fcn.Decoder_LaneNet_Interplt()

    def forward(self, input):
        x = self.encoder.forward(input)

        input_tensor_list = [self.encoder.c1, self.encoder.c2, self.encoder.c3, self.encoder.c4, x]
        embedding= self.decoder_embed.forward(input_tensor_list)
        logit = self.decoder_logit.forward(input_tensor_list)

        return embedding, logit


class LaneNet_ENet_1E2D(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = ENet_encoder()
        self.decoder_logit = ENet_decoder(2)
        self.decoder_embed = ENet_decoder(3)

    def forward(self, input):
        x, max_indices1, max_indices2 = self.encoder.forward(input)

        logit = self.decoder_logit(x, max_indices1, max_indices2)
        embedding = self.decoder_embed(x, max_indices1, max_indices2)

        return embedding, logit


class LaneNet_ENet_1E1D(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = ENet(64)
        self.conv_logit = nn.Conv2d(64, 2, 1, 1)
        self.conv_embedding = nn.Conv2d(64, 3, 1, 1)

    def forward(self, input):
        x = self.model.forward(input)

        logit = self.conv_logit(x)
        embedding = self.conv_embedding(x)

        return embedding, logit


class LaneNet_ICNet_1E2D(ICNet_1E2D):

    def __init__(self):
        super().__init__()


class LaneNet_ICNet_1E1D(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = ICNet(64)
        self.conv_logit = nn.Conv2d(64, 2, 1, 1)
        self.conv_embedding = nn.ConvTranspose2d(64, 3, 1, 1)

    def forward(self, input):
        x = self.model.forward(input)

        logit = self.conv_logit(x)
        embedding = self.conv_embedding(x)

        return embedding, logit


if __name__ == '__main__':
    img = torch.randn(1, 3, 512, 512)
    model = LaneNet_ENet_1E2D()
    embedding, logit = model(img)

    print(embedding)
    print(embedding.shape)
