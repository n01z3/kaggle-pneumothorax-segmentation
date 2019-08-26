import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision import models
import pretrainedmodels


from legacy.pretrained_models import se_resnet50, se_resnext50_32x4d, se_resnext101_32x4d, senet154
import legacy.blocks as blocks



## https://github.com/ternaus/robot-surgery-segmentation/blob/master/models.py ##


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int):
        super(ConvRelu, self).__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    """
    Paramaters for Deconvolution were chosen to avoid artifacts, following
    link https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear"),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)


class UNet11(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network used
            True - encoder pre-trained with VGG11
        """
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.num_classes = num_classes

        self.encoder = models.vgg11(pretrained=pretrained).features

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(self.encoder[0], self.relu)

        self.conv2 = nn.Sequential(self.encoder[3], self.relu)

        self.conv3 = nn.Sequential(self.encoder[6], self.relu, self.encoder[8], self.relu)
        self.conv4 = nn.Sequential(self.encoder[11], self.relu, self.encoder[13], self.relu)

        self.conv5 = nn.Sequential(self.encoder[16], self.relu, self.encoder[18], self.relu)

        self.center = DecoderBlock(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv=True)
        self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv=True)
        self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 4, is_deconv=True)
        self.dec3 = DecoderBlock(256 + num_filters * 4, num_filters * 4 * 2, num_filters * 2, is_deconv=True)
        self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 2 * 2, num_filters, is_deconv=True)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)

        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)
        self.final_sigm = nn.Sequential(nn.Conv2d(num_filters, num_classes, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))
        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec1), dim=1)
        else:
            x_out = self.final_sigm(dec1)

        return x_out


class UNet16(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network used
            True - encoder pre-trained with VGG11
        """
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.vgg16(pretrained=pretrained).features

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder[0], self.relu, self.encoder[2], self.relu)

        self.conv2 = nn.Sequential(self.encoder[5], self.relu, self.encoder[7], self.relu)

        self.conv3 = nn.Sequential(
            self.encoder[10], self.relu, self.encoder[12], self.relu, self.encoder[14], self.relu
        )

        self.conv4 = nn.Sequential(
            self.encoder[17], self.relu, self.encoder[19], self.relu, self.encoder[21], self.relu
        )

        self.conv5 = nn.Sequential(
            self.encoder[24], self.relu, self.encoder[26], self.relu, self.encoder[28], self.relu
        )

        self.center = DecoderBlock(512, num_filters * 8 * 2, num_filters * 8)

        self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec3 = DecoderBlock(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 2 * 2, num_filters)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

        self.final_sigm = nn.Sequential(nn.Conv2d(num_filters, num_classes, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec1), dim=1)
        else:
            x_out = self.final_sigm(dec1)

        return x_out


class DecoderBlockLinkNet(nn.Module):
    def __init__(self, in_channels, n_filters):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)

        # B, C/4, H, W -> B, C/4, 2 * H, 2 * W
        self.deconv2 = nn.ConvTranspose2d(
            in_channels // 4, in_channels // 4, kernel_size=4, stride=2, padding=1, output_padding=0
        )
        self.norm2 = nn.BatchNorm2d(in_channels // 4)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.norm1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        # x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        # x = self.norm3(x)
        x = self.relu(x)
        return x


class LinkNet34(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True):
        super().__init__()
        assert num_channels == 3
        self.num_classes = num_classes
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=pretrained)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.decoder4 = DecoderBlockLinkNet(filters[3], filters[2])
        self.decoder3 = DecoderBlockLinkNet(filters[2], filters[1])
        self.decoder2 = DecoderBlockLinkNet(filters[1], filters[0])
        self.decoder1 = DecoderBlockLinkNet(filters[0], filters[0])

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

        self.final = nn.Sequential(nn.Conv2d(32, num_classes, 2, padding=1), nn.Sigmoid())

    # noinspection PyCallingNonCallable
    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with Skip Connections
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        # f5 = self.finalconv3(f4)

        if self.num_classes > 1:
            f5 = self.finalconv3(f4)
            x_out = F.log_softmax(f5, dim=1)
        else:
            f5 = self.final(f4)
            x_out = f5
        return x_out


class AlbuNet(nn.Module):
    """
        UNet (https://arxiv.org/abs/1505.04597) with Resnet34(https://arxiv.org/abs/1512.03385) encoder
        Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/
        """

    def __init__(self, num_classes=1, num_filters=32, pretrained=False, is_deconv=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet34
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.resnet34(pretrained=pretrained)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu, self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.center = DecoderBlock(512, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlock(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlock(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlock(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlock(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)
        self.final_sigm = nn.Sequential(nn.Conv2d(num_filters, num_classes, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec0), dim=1)
        else:
            x_out = self.final_sigm(dec0)

        return x_out


class LinkNet50(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True):
        super().__init__()
        assert num_channels == 3
        self.num_classes = num_classes
        filters = [4 * 64, 4 * 128, 4 * 256, 4 * 512]
        resnet = models.resnet50(pretrained=pretrained)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.decoder4 = DecoderBlockLinkNet(filters[3], filters[2])
        self.decoder3 = DecoderBlockLinkNet(filters[2], filters[1])
        self.decoder2 = DecoderBlockLinkNet(filters[1], filters[0])
        self.decoder1 = DecoderBlockLinkNet(filters[0], filters[0])

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

        self.final = nn.Sequential(nn.Conv2d(32, num_classes, 2, padding=1), nn.Sigmoid())

    # noinspection PyCallingNonCallable
    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with Skip Connections
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        # f5 = self.finalconv3(f4)

        if self.num_classes > 1:
            f5 = self.finalconv3(f4)
            x_out = F.log_softmax(f5, dim=1)
        else:
            f5 = self.final(f4)
            x_out = f5
        return x_out


class SELinkNet50(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True):
        super().__init__()
        assert num_channels == 3
        self.num_classes = num_classes
        filters = [4 * 64, 4 * 128, 4 * 256, 4 * 512]
        resnet = se_resnet50()

        # self.firstconv = resnet.conv1
        # self.firstbn = resnet.bn1
        # self.firstrelu = resnet.relu
        # self.firstmaxpool = resnet.maxpool

        self.encoder0 = resnet.layer0
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.decoder4 = DecoderBlockLinkNet(filters[3], filters[2])
        self.decoder3 = DecoderBlockLinkNet(filters[2], filters[1])
        self.decoder2 = DecoderBlockLinkNet(filters[1], filters[0])
        self.decoder1 = DecoderBlockLinkNet(filters[0], filters[0])

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

        self.final = nn.Sequential(nn.Conv2d(32, num_classes, 2, padding=1), nn.Sigmoid())

    # noinspection PyCallingNonCallable
    def forward(self, x):
        # Encoder
        # x = self.firstconv(x)
        # x = self.firstbn(x)
        # x = self.firstrelu(x)
        # x = self.firstmaxpool(x)
        x = self.encoder0(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with Skip Connections
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        # f5 = self.finalconv3(f4)

        if self.num_classes > 1:
            f5 = self.finalconv3(f4)
            x_out = F.log_softmax(f5, dim=1)
        else:
            f5 = self.final(f4)
            x_out = f5
        return x_out


class SELinkNet50_multiout(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True):
        super().__init__()
        assert num_channels == 3
        self.num_classes = num_classes
        filters = [4 * 64, 4 * 128, 4 * 256, 4 * 512]
        resnet = se_resnet50()

        self.encoder0 = resnet.layer0
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.cam_gap = nn.AdaptiveAvgPool2d((1, 1))
        self.cam_lin1 = nn.Linear(2048, 100)
        # self.cam_lin1a = nn.Linear(500,100)
        self.cam_lin2 = nn.Linear(100, self.num_classes)  # 2 - lesion and background [0,1]?

    # noinspection PyCallingNonCallable
    def forward(self, x):
        x = self.encoder0(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        e4_gap = self.cam_gap(e4)
        e4_gap = e4_gap.view(e4_gap.size(0), -1)
        e4_lin = self.cam_lin1(e4_gap)
        e4_lin2 = self.cam_lin2(e4_lin)

        x_out = F.log_softmax(e4_lin2, dim=1)
        # x_out = F.softmax(e4_lin2)

        return x_out


class SEResNext50(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True):
        super().__init__()
        assert num_channels == 3
        self.num_classes = num_classes
        filters = [4 * 64, 4 * 128, 4 * 256, 4 * 512]
        resnet = se_resnext50_32x4d(drop_rate=0.2)

        self.encoder0 = resnet.layer0
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.cam_gap = nn.AdaptiveAvgPool2d((1, 1))
        self.cam_lin1 = nn.Linear(2048, 100)
        # self.cam_lin1a = nn.Linear(500,100)
        self.cam_lin2 = nn.Linear(100, self.num_classes)  # 2 - lesion and background [0,1]?

    # noinspection PyCallingNonCallable
    def forward(self, x):
        x = self.encoder0(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        e4_gap = self.cam_gap(e4)
        e4_gap = e4_gap.view(e4_gap.size(0), -1)
        e4_lin = self.cam_lin1(e4_gap)
        e4_lin2 = self.cam_lin2(e4_lin)

        x_out = F.log_softmax(e4_lin2, dim=1)
        # x_out = F.softmax(e4_lin2)

        return x_out


class UnetSEResNext50(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True):
        super().__init__()
        assert num_channels == 3
        self.num_classes = num_classes
        filters = [4 * 64, 4 * 128, 4 * 256, 4 * 512]
        resnet = se_resnext50_32x4d()

        self.encoder0 = resnet.layer0
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.decoder4 = DecoderBlockLinkNet(filters[3], filters[2])
        self.decoder3 = DecoderBlockLinkNet(filters[2], filters[1])
        self.decoder2 = DecoderBlockLinkNet(filters[1], filters[0])
        self.decoder1 = DecoderBlockLinkNet(filters[0], filters[0])

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

        self.final = nn.Sequential(nn.Conv2d(32, num_classes, 2, padding=1), nn.Sigmoid())

    # noinspection PyCallingNonCallable
    def forward(self, x):
        # Encoder
        # x = self.firstconv(x)
        # x = self.firstbn(x)
        # x = self.firstrelu(x)
        # x = self.firstmaxpool(x)
        x = self.encoder0(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with Skip Connections
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        # f5 = self.finalconv3(f4)

        if self.num_classes > 1:
            f5 = self.finalconv3(f4)
            x_out = F.log_softmax(f5, dim=1)
        else:
            f5 = self.final(f4)
            x_out = f5
        return x_out


class UnetSEResNext101(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True):
        super().__init__()
        assert num_channels == 3
        self.num_classes = num_classes
        filters = [4 * 64, 4 * 128, 4 * 256, 4 * 512]
        resnet = se_resnext101_32x4d(drop_rate=0.2)

        self.encoder0 = resnet.layer0
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.decoder4 = DecoderBlockLinkNet(filters[3], filters[2])
        self.decoder3 = DecoderBlockLinkNet(filters[2], filters[1])
        self.decoder2 = DecoderBlockLinkNet(filters[1], filters[0])
        self.decoder1 = DecoderBlockLinkNet(filters[0], filters[0])

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

        self.final = nn.Sequential(nn.Conv2d(32, num_classes, 2, padding=1), nn.Sigmoid())

    # noinspection PyCallingNonCallable
    def forward(self, x):
        # Encoder
        # x = self.firstconv(x)
        # x = self.firstbn(x)
        # x = self.firstrelu(x)
        # x = self.firstmaxpool(x)
        x = self.encoder0(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with Skip Connections
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        # f5 = self.finalconv3(f4)

        if self.num_classes > 1:
            f5 = self.finalconv3(f4)
            x_out = F.log_softmax(f5, dim=1)
        else:
            f5 = self.final(f4)
            x_out = f5
        return x_out


class LinknetSEResNext152(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True):
        super().__init__()
        assert num_channels == 3
        self.num_classes = num_classes
        filters = [4 * 64, 4 * 128, 4 * 256, 4 * 512]
        resnet = se_resnet152(drop_rate=0.2)

        self.encoder0 = resnet.layer0
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.decoder4 = DecoderBlockLinkNet(filters[3], filters[2])
        self.decoder3 = DecoderBlockLinkNet(filters[2], filters[1])
        self.decoder2 = DecoderBlockLinkNet(filters[1], filters[0])
        self.decoder1 = DecoderBlockLinkNet(filters[0], filters[0])

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

        self.final = nn.Sequential(nn.Conv2d(32, num_classes, 2, padding=1), nn.Sigmoid())

    # noinspection PyCallingNonCallable
    def forward(self, x):
        # Encoder
        # x = self.firstconv(x)
        # x = self.firstbn(x)
        # x = self.firstrelu(x)
        # x = self.firstmaxpool(x)
        x = self.encoder0(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with Skip Connections
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        # f5 = self.finalconv3(f4)

        if self.num_classes > 1:
            f5 = self.finalconv3(f4)
            x_out = F.log_softmax(f5, dim=1)
        else:
            f5 = self.final(f4)
            x_out = f5
        return x_out


class UnetSEResNext101_logits(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True):
        super().__init__()
        assert num_channels == 3
        self.num_classes = num_classes
        filters = [4 * 64, 4 * 128, 4 * 256, 4 * 512]
        resnet = se_resnext101_32x4d(drop_rate=0.2)

        self.encoder0 = resnet.layer0
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.decoder4 = DecoderBlockLinkNet(filters[3], filters[2])
        self.decoder3 = DecoderBlockLinkNet(filters[2], filters[1])
        self.decoder2 = DecoderBlockLinkNet(filters[1], filters[0])
        self.decoder1 = DecoderBlockLinkNet(filters[0], filters[0])

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

        self.final = nn.Sequential(nn.Conv2d(32, num_classes, 2, padding=1))

    # noinspection PyCallingNonCallable
    def forward(self, x):
        # Encoder
        # x = self.firstconv(x)
        # x = self.firstbn(x)
        # x = self.firstrelu(x)
        # x = self.firstmaxpool(x)
        x = self.encoder0(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with Skip Connections
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        # f5 = self.finalconv3(f4)

        if self.num_classes > 1:
            f5 = self.finalconv3(f4)
            x_out = F.log_softmax(f5, dim=1)
        else:
            f5 = self.final(f4)
            x_out = f5
        return x_out


class UnetSENet154(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True):
        super().__init__()
        assert num_channels == 3
        self.num_classes = num_classes
        filters = [4 * 64, 4 * 128, 4 * 256, 4 * 512]
        resnet = senet154()

        self.encoder0 = resnet.layer0
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.decoder4 = DecoderBlockLinkNet(filters[3], filters[2])
        self.decoder3 = DecoderBlockLinkNet(filters[2], filters[1])
        self.decoder2 = DecoderBlockLinkNet(filters[1], filters[0])
        self.decoder1 = DecoderBlockLinkNet(filters[0], filters[0])

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

        self.final = nn.Sequential(nn.Conv2d(32, num_classes, 2, padding=1), nn.Sigmoid())

    # noinspection PyCallingNonCallable
    def forward(self, x):
        # Encoder
        # x = self.firstconv(x)
        # x = self.firstbn(x)
        # x = self.firstrelu(x)
        # x = self.firstmaxpool(x)
        x = self.encoder0(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with Skip Connections
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        # f5 = self.finalconv3(f4)

        if self.num_classes > 1:
            f5 = self.finalconv3(f4)
            x_out = F.log_softmax(f5, dim=1)
        else:
            f5 = self.final(f4)
            x_out = f5
        return x_out


class UNetResNextHyperSE(nn.Module):
    """PyTorch U-Net model using ResNet(34, 101 or 152) encoder.
    UNet: https://arxiv.org/abs/1505.04597
    ResNext: https://arxiv.org/abs/1611.05431
    Args:
    encoder_depth (int): Depth of a ResNet encoder (34, 101 or 152).
    num_classes (int): Number of output classes.
    num_filters (int, optional): Number of filters in the last layer of decoder. Defaults to 32.
    dropout_2d (float, optional): Probability factor of dropout layer before output layer. Defaults to 0.2.
    is_deconv (bool, optional):
        False: bilinear interpolation is used in decoder.
        True: deconvolution is used in decoder.
        Defaults to False.
    """

    def __init__(self, encoder_depth, num_classes, num_filters=32, dropout_2d=0.2, is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d

        if encoder_depth == 50:
            self.encoder = pretrainedmodels.__dict__['se_resnext50_32x4d'](num_classes=1000, pretrained='imagenet')
            bottom_channel_nr = 2048
        elif encoder_depth == 101:
            self.encoder = pretrainedmodels.__dict__['se_resnext101_32x4d'](num_classes=1000, pretrained='imagenet')
            bottom_channel_nr = 2048
        else:
            raise NotImplementedError('only 34, 101, 152 version of Resnet are implemented')

        self.pool = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU(inplace=True)

        conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
        conv1.weight = self.encoder.layer0.conv1.weight

        self.input_adjust = blocks.EncoderBlock(
            nn.Sequential(conv1, self.encoder.layer0.bn1, self.encoder.layer0.relu1, self.pool),
            num_filters*2
        )

        self.conv1 = blocks.EncoderBlock(self.encoder.layer1, bottom_channel_nr//8)
        self.conv2 = blocks.EncoderBlock(self.encoder.layer2, bottom_channel_nr//4)
        self.conv3 = blocks.EncoderBlock(self.encoder.layer3, bottom_channel_nr//2)
        self.conv4 = blocks.EncoderBlock(self.encoder.layer4, bottom_channel_nr)

        self.dec4 = blocks.DecoderBlockV4(bottom_channel_nr, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = blocks.DecoderBlockV4(bottom_channel_nr // 2 + num_filters * 8, num_filters * 8 * 2, num_filters * 8,
                                   is_deconv)
        self.dec2 = blocks.DecoderBlockV4(bottom_channel_nr // 4 + num_filters * 8, num_filters * 4 * 2, num_filters * 2,
                                   is_deconv)
        self.dec1 = blocks.DecoderBlockV4(bottom_channel_nr // 8 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2,
                                   is_deconv)
        self.final = nn.Sequential(nn.Conv2d(2 * 2, num_classes, kernel_size=1),
                        nn.Sigmoid(),)

        # deep supervision
        self.dsv4 = blocks.UnetDsv3(in_size=num_filters * 8, out_size=num_classes, scale_factor=8 * 2)
        self.dsv3 = blocks.UnetDsv3(in_size=num_filters * 8, out_size=num_classes, scale_factor=4 * 2)
        self.dsv2 = blocks.UnetDsv3(in_size=num_filters * 2, out_size=num_classes, scale_factor=2 * 2)
        self.dsv1 = nn.Conv2d(in_channels=num_filters * 2 * 2, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        input_adjust = self.input_adjust(x)
        conv1 = self.conv1(input_adjust)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        center = self.conv4(conv3)

        dec4 = self.dec4(center)
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        # Deep Supervision
        dsv4 = self.dsv4(dec4)
        dsv3 = self.dsv3(dec3)
        dsv2 = self.dsv2(dec2)
        dsv1 = self.dsv1(dec1)
        dsv0 = torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1)

        return self.final(dsv0)

def get_hypermodel(model_type, **model_pars):
    """
    Create a new hyper model
    :param model_type: str, type of model, one of ['UNet11', 'UNet16', 'LinkNet34', 'UNet']
    :param model_pars: dict, initialization parameters for the model
    :return: instance of nn.Module, a new Pytorch model
    """

    if model_type == 'UNetResNextHyperSE50':
        return UNetResNextHyperSE(encoder_depth=50, num_classes=1, num_filters=32, is_deconv=True, dropout_2d=0.2)

    if model_type == 'UNetResNextHyperSE101':
        return UNetResNextHyperSE(encoder_depth=101, num_classes=1, num_filters=32, is_deconv=True, dropout_2d=0.2)

    raise ValueError('Unknown model type: {}'.format(model_type))