import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import torch.utils.model_zoo as model_zoo


class DecoderBlock(nn.Module):
    def __init__(self, ch_in, ch_mid, ch_out, k, s, p, op=0):
        super(DecoderBlock, self).__init__()

        '''Transposed Conv Block'''
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_mid, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=ch_mid, out_channels=ch_out, kernel_size=k,
                               stride=s, padding=p, output_padding=op),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Decoder_LaneNet_TConv(nn.Module):

    def __init__(self):
        super(Decoder_LaneNet_TConv, self).__init__()

        '''Instance Segmentation Method'''
        self.dec5 = DecoderBlock(ch_in=512, ch_mid=64, ch_out=256, k=4, s=2, p=1)  # for 512x288
        # self.dec5 = DecoderBlock(ch_in=512, ch_mid=64, ch_out=256, k=4, s=2, p=(2, 1), op=(1, 0))  # for 1280x720
        self.dec4 = DecoderBlock(ch_in=256 * 2, ch_mid=64, ch_out=128, k=4, s=2, p=1)
        self.dec3 = DecoderBlock(ch_in=128 * 2, ch_mid=64, ch_out=64, k=4, s=2, p=1)
        self.dec2 = DecoderBlock(ch_in=64 * 2, ch_mid=64, ch_out=64, k=4, s=2, p=1)
        self.dec1 = DecoderBlock(ch_in=64 * 2, ch_mid=64, ch_out=64, k=4, s=2, p=1)
        self.conv_embedding = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1)
        self.conv_logit = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, input_tensor_list):
        """
        :param input_tensor_list:
        :return:
        """

        dec5 = self.dec5(input_tensor_list[4])
        dec4 = self.dec4(torch.cat((dec5, input_tensor_list[3]), 1))
        dec3 = self.dec3(torch.cat((dec4, input_tensor_list[2]), 1))
        dec2 = self.dec2(torch.cat((dec3, input_tensor_list[1]), 1))
        dec1 = self.dec2(torch.cat((dec2, input_tensor_list[0]), 1))

        embedding = self.conv_embedding(dec1)
        logit = self.conv_logit(dec1)

        return embedding, logit


class Decoder_LaneNet_TConv_Embed(nn.Module):

    def __init__(self):
        super(Decoder_LaneNet_TConv_Embed, self).__init__()

        '''Instance Segmentation Method'''
        self.dec5 = DecoderBlock(ch_in=512, ch_mid=64, ch_out=256, k=4, s=2, p=1)  # for 512x288
        # self.dec5 = DecoderBlock(ch_in=512, ch_mid=64, ch_out=256, k=4, s=2, p=(2, 1), op=(1, 0))  # for 1280x720
        self.dec4 = DecoderBlock(ch_in=256 * 2, ch_mid=64, ch_out=128, k=4, s=2, p=1)
        self.dec3 = DecoderBlock(ch_in=128 * 2, ch_mid=64, ch_out=64, k=4, s=2, p=1)
        self.dec2 = DecoderBlock(ch_in=64 * 2, ch_mid=64, ch_out=64, k=4, s=2, p=1)
        self.dec1 = DecoderBlock(ch_in=64 * 2, ch_mid=64, ch_out=64, k=4, s=2, p=1)
        self.conv_embedding = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1)  # embedding dim

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, input_tensor_list):
        """
        :param input_tensor_list:
        :return:
        """

        dec5 = self.dec5(input_tensor_list[4])
        dec4 = self.dec4(torch.cat((dec5, input_tensor_list[3]), 1))
        dec3 = self.dec3(torch.cat((dec4, input_tensor_list[2]), 1))
        dec2 = self.dec2(torch.cat((dec3, input_tensor_list[1]), 1))
        dec1 = self.dec1(torch.cat((dec2, input_tensor_list[0]), 1))

        embedding = self.conv_embedding(dec1)

        return embedding


class Decoder_LaneNet_TConv_Logit(nn.Module):

    def __init__(self):
        super(Decoder_LaneNet_TConv_Logit, self).__init__()

        '''Instance Segmentation Method'''
        self.dec5 = DecoderBlock(ch_in=512, ch_mid=64, ch_out=256, k=4, s=2, p=1)  # for 512x288
        # self.dec5 = DecoderBlock(ch_in=512, ch_mid=64, ch_out=256, k=4, s=2, p=(2, 1), op=(1, 0))  # for 1280x720
        self.dec4 = DecoderBlock(ch_in=256 * 2, ch_mid=64, ch_out=128, k=4, s=2, p=1)
        self.dec3 = DecoderBlock(ch_in=128 * 2, ch_mid=64, ch_out=64, k=4, s=2, p=1)
        self.dec2 = DecoderBlock(ch_in=64 * 2, ch_mid=64, ch_out=64, k=4, s=2, p=1)
        self.dec1 = DecoderBlock(ch_in=64 * 2, ch_mid=64, ch_out=64, k=4, s=2, p=1)
        self.conv_logit = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, input_tensor_list):
        """
        :param input_tensor_list:
        :return:
        """

        dec5 = self.dec5(input_tensor_list[4])
        dec4 = self.dec4(torch.cat((dec5, input_tensor_list[3]), 1))
        dec3 = self.dec3(torch.cat((dec4, input_tensor_list[2]), 1))
        dec2 = self.dec2(torch.cat((dec3, input_tensor_list[1]), 1))
        dec1 = self.dec1(torch.cat((dec2, input_tensor_list[0]), 1))

        logit = self.conv_logit(dec1)

        return logit


class Decoder_LaneNet_Interplt(nn.Module):

    def __init__(self):
        super(Decoder_LaneNet_Interplt, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        channel = 64
        
        # Conv layers used to fuse feature map from different levels
        self.conv_c5 = nn.Conv2d(512, channel, 1)
        self.conv_c4 = nn.Conv2d(256, channel, 1)
        self.conv_c3 = nn.Conv2d(128, channel, 1)
        self.conv_c2 = nn.Conv2d(64, channel, 1)
        self.conv_c1 = nn.Conv2d(64, channel, 1)
        
        # Three types of decoder architectures to be plugged in
        '''1- conv1x1'''
        # self.conv_d5 = nn.Conv2d(channel, channel, 1)
        # self.conv_d4 = nn.Conv2d(2 * channel, channel, 1)
        # self.conv_d3 = nn.Conv2d(2 * channel, channel, 1)
        # self.conv_d2 = nn.Conv2d(2 * channel, channel, 1)
        # self.conv_d1 = nn.Conv2d(2 * channel, channel, 1)

        '''2- conv3x3'''
        self.conv_d5 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_d4 = nn.Conv2d(2 * channel, channel, 3, padding=1)
        self.conv_d3 = nn.Conv2d(2 * channel, channel, 3, padding=1)
        self.conv_d2 = nn.Conv2d(2 * channel, channel, 3, padding=1)
        self.conv_d1 = nn.Conv2d(2 * channel, channel, 3, padding=1)

        '''3- conv3x3_& conv1x1'''
        # self.conv_d5 = nn.Conv2d(channel, channel, 1)
        # self.conv_d4 = nn.Conv2d(2 * channel, channel, 1)
        # self.conv_d3 = nn.Conv2d(2 * channel, channel, 1)
        # self.conv_d2 = nn.Conv2d(2 * channel, channel, 3, padding=1)
        # self.conv_d1 = nn.Conv2d(2 * channel, channel, 3, padding=1)

        self.conv_embedding = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1)  # embedding dim
        # self.conv_embedding = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, padding=0)  # embedding dim
        self.conv_logit = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, input_tensor_list):
        """
        :param input_tensor_list:
        :return:
        """
        c1, c2, c3, c4, c5 = input_tensor_list

        d5 = self.relu(self.conv_c5(c5))

        d4 = torch.cat((self.conv_d5(F.interpolate(d5, scale_factor=2, mode='bilinear')),
                       self.relu(self.conv_c4(c4))), 1)
        d3 = torch.cat((self.conv_d4(F.interpolate(d4, scale_factor=2, mode='bilinear')),
                       self.relu(self.conv_c3(c3))), 1)
        d2 = torch.cat((self.conv_d3(F.interpolate(d3, scale_factor=2, mode='bilinear')),
                       self.relu(self.conv_c2(c2))), 1)
        d1 = torch.cat((self.conv_d2(F.interpolate(d2, scale_factor=2, mode='bilinear')),
                       self.relu(self.conv_c1(c1))), 1)
        d0 = self.relu(self.conv_d1(F.interpolate(d1, scale_factor=2, mode='bilinear')))

        embedding = self.conv_embedding(d0)
        logit = self.conv_logit(d0)

        return embedding, logit
