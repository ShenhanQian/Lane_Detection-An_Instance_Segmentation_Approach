"""Image Cascade Network"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.resnetv1b import resnet50_v1s


class SegBaseModel(nn.Module):
    r"""Base Model for Semantic Segmentation
    Parameters
    ----------
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    """

    def __init__(self, nclass, aux, backbone='resnet50', jpu=False, pretrained_base=True, **kwargs):
        super(SegBaseModel, self).__init__()
        dilated = False if jpu else True
        self.aux = aux
        self.nclass = nclass
        if backbone == 'resnet50':
            self.pretrained = resnet50_v1s(pretrained=pretrained_base, dilated=dilated, **kwargs)
        # elif backbone == 'resnet101':
        #     self.pretrained = resnet101_v1s(pretrained=pretrained_base, dilated=dilated, **kwargs)
        # elif backbone == 'resnet152':
        #     self.pretrained = resnet152_v1s(pretrained=pretrained_base, dilated=dilated, **kwargs)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

        # self.jpu = JPU([512, 1024, 2048], width=512, **kwargs) if jpu else None

    def base_forward(self, x):
        """forwarding pre-trained network"""
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)

        return c1, c2, c3, c4

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)[0]

    def demo(self, x):
        pred = self.forward(x)
        if self.aux:
            pred = pred[0]
        return pred


class ICNet(SegBaseModel):
    """Image Cascade Network"""

    def __init__(self, nclass, backbone='resnet50', aux=False, jpu=False, pretrained_base=True, **kwargs):
        super(ICNet, self).__init__(nclass, aux, backbone, pretrained_base=pretrained_base, **kwargs)
        self.conv_sub1 = nn.Sequential(
            _ConvBNReLU(3, 32, 3, 2, **kwargs),
            _ConvBNReLU(32, 32, 3, 2, **kwargs),
            _ConvBNReLU(32, 64, 3, 2, **kwargs)
        )

        self.head = _ICHead(nclass, **kwargs)

        self.__setattr__('exclusive', ['conv_sub1', 'head'])

    def forward(self, x):
        # sub 1
        x_sub1 = self.conv_sub1(x)

        # sub 2
        x_sub2 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        _, x_sub2, _, _ = self.base_forward(x_sub2)

        # sub 4
        x_sub4 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=True)
        _, _, _, x_sub4 = self.base_forward(x_sub4)

        outputs = self.head(x_sub1, x_sub2, x_sub4)

        return outputs[0]


class ICNet_1E2D(SegBaseModel):
    """Image Cascade Network"""

    def __init__(self, backbone='resnet50', aux=False, jpu=False, pretrained_base=True, **kwargs):
        super(ICNet_1E2D, self).__init__(aux, backbone, pretrained_base=pretrained_base, **kwargs)
        self.conv_sub1 = nn.Sequential(
            _ConvBNReLU(3, 32, 3, 2, **kwargs),
            _ConvBNReLU(32, 32, 3, 2, **kwargs),
            _ConvBNReLU(32, 64, 3, 2, **kwargs)
        )

        self.head_logit = _ICHead(2, **kwargs)
        self.head_embed = _ICHead(3, **kwargs)

        self.__setattr__('exclusive', ['conv_sub1', 'head'])

    def forward(self, x):
        # sub 1
        x_sub1 = self.conv_sub1(x)

        # sub 2
        x_sub2 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        _, x_sub2, _, _ = self.base_forward(x_sub2)

        # sub 4
        x_sub4 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=True)
        _, _, _, x_sub4 = self.base_forward(x_sub4)

        outputs_logit = self.head_logit(x_sub1, x_sub2, x_sub4)
        outputs_embed = self.head_embed(x_sub1, x_sub2, x_sub4)

        return outputs_embed[0], outputs_logit[0]


class _ICHead(nn.Module):
    def __init__(self, nclass, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_ICHead, self).__init__()
        self.cff_12 = CascadeFeatureFusion(512, 64, 128, nclass, norm_layer, **kwargs)
        self.cff_24 = CascadeFeatureFusion(2048, 512, 128, nclass, norm_layer, **kwargs)

        self.conv_cls = nn.Conv2d(128, nclass, 1, bias=False)

    def forward(self, x_sub1, x_sub2, x_sub4):
        outputs = list()
        x_cff_24, x_24_cls = self.cff_24(x_sub4, x_sub2)
        outputs.append(x_24_cls)
        x_cff_12, x_12_cls = self.cff_12(x_sub2, x_sub1)
        outputs.append(x_12_cls)

        up_x2 = F.interpolate(x_cff_12, scale_factor=2, mode='bilinear', align_corners=True)
        up_x2 = self.conv_cls(up_x2)
        outputs.append(up_x2)
        up_x8 = F.interpolate(up_x2, scale_factor=4, mode='bilinear', align_corners=True)
        outputs.append(up_x8)
        # 1 -> 1/4 -> 1/8 -> 1/16
        outputs.reverse()

        return outputs


class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                 groups=1, norm_layer=nn.BatchNorm2d, bias=False, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CascadeFeatureFusion(nn.Module):
    """CFF Unit"""

    def __init__(self, low_channels, high_channels, out_channels, nclass, norm_layer=nn.BatchNorm2d, **kwargs):
        super(CascadeFeatureFusion, self).__init__()
        self.conv_low = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, 3, padding=2, dilation=2, bias=False),
            norm_layer(out_channels)
        )
        self.conv_high = nn.Sequential(
            nn.Conv2d(high_channels, out_channels, 1, bias=False),
            norm_layer(out_channels)
        )
        self.conv_low_cls = nn.Conv2d(out_channels, nclass, 1, bias=False)

    def forward(self, x_low, x_high):
        x_low = F.interpolate(x_low, size=x_high.size()[2:], mode='bilinear', align_corners=True)
        x_low = self.conv_low(x_low)
        x_high = self.conv_high(x_high)
        x = x_low + x_high
        x = F.relu(x, inplace=True)
        x_low_cls = self.conv_low_cls(x_low)

        return x, x_low_cls


if __name__ == '__main__':
    img = torch.randn(1, 3, 512, 512)
    model = ICNet(3, backbone='resnet50', pretrained_base=True)

    outputs = model(img)

