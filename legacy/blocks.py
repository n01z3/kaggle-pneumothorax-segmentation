from torch import nn
import torch
from torch.nn import functional as F


class ChannelAttentionGate(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttentionGate, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y


class SpatialAttentionGate(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SpatialAttentionGate, self).__init__()
        self.fc1 = nn.Conv2d(channel, reduction, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(reduction, 1, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x


class EncoderBlock(nn.Module):
    def __init__(self, block, out_channels):
        super(EncoderBlock, self).__init__()
        self.block = block
        self.out_channels = out_channels
        self.spatial_gate = SpatialAttentionGate(out_channels)
        self.channel_gate = ChannelAttentionGate(out_channels)

    def forward(self, x):
        x = self.block(x)
        g1 = self.spatial_gate(x)
        g2 = self.channel_gate(x)

        return x*g1 + x*g2


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True)
                                  )

    def forward(self, x):
        return self.conv(x)


class UnetDsv3(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(UnetDsv3, self).__init__()
        self.dsv = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=4, stride=2, padding=1),
                                 nn.Upsample(scale_factor=scale_factor, mode='bilinear'), )

    def forward(self, input):
        return self.dsv(input)


class DecoderBlockV4(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV4, self).__init__()
        self.is_deconv = is_deconv

        self.deconv = nn.Sequential(
            ConvBnRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Sequential(
            ConvBnRelu(in_channels, out_channels),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )

        self.spatial_gate = SpatialAttentionGate(out_channels)
        self.channel_gate = ChannelAttentionGate(out_channels)

    def forward(self, x):
        if self.is_deconv:
            x = self.deconv(x)
        else:
            x = self.upsample(x)

        g1 = self.spatial_gate(x)
        g2 = self.channel_gate(x)
        x = x*g1 + x*g2

        return x


class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV2, self).__init__()
        self.is_deconv = is_deconv

        self.deconv = nn.Sequential(
            ConvBnRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Sequential(
            ConvBnRelu(in_channels, out_channels),
            nn.Upsample(scale_factor=2, mode='bilinear'),

        )

    def forward(self, x):
        if self.is_deconv:
            x = self.deconv(x)
        else:
            x = self.upsample(x)
        return x