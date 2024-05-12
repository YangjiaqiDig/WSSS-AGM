import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from network.attention import Block


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNetWeakly(nn.Module):
    def __init__(
        self,
        n_classes,
        n_classes_layer=2,
        n_channels=3,
        n_layer_channels=3,
        bilinear=False,
        is_size=(224, 224),
        include_seg_cls=False,
    ):
        super(UNetWeakly, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_classes_layer = n_classes_layer
        self.bilinear = bilinear
        self.is_size = is_size
        self.include_seg_cls = include_seg_cls
        factor = 2 if bilinear else 1
        self.feature_maps = {}

        self.inc_orig = DoubleConv(n_channels, 64)
        self.down1_orig = Down(64, 128)
        self.down2_orig = Down(128, 256)
        self.down3_orig = Down(256, 512)
        self.down4_orig = Down(512, 1024 // factor)

        self.inc_layer = DoubleConv(
            n_layer_channels, 64
        )  # might need change for layers (binarilized better?)
        self.down1_layer = Down(64, 128)
        self.down2_layer = Down(128, 256)
        self.down3_layer = Down(256, 512)
        self.down4_layer = Down(512, 1024 // factor)

        self.attn1 = Block(
            dim=512,
            num_heads=8,
            mlp_ratio=4,
            qkv_bias=True,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
        )
        self.attn2 = Block(
            dim=1024,
            num_heads=8,
            mlp_ratio=4,
            qkv_bias=True,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
        )

        self.cls_orig = OutConv(1024, n_classes)
        self.cls_layer = OutConv(1024, n_classes_layer)

        # self.up1 = Up(1024, 512 // factor, bilinear)
        # self.up2 = Up(512, 256 // factor, bilinear)
        # self.up3 = Up(256, 128 // factor, bilinear)
        # self.up4 = Up(128, 64, bilinear)

        self.outc = OutConv(64, n_classes)
        # self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

    def cls_head_orig(self, feature):
        y = self.cls_orig(feature)  # x [b, 1024, 16, 16], y [b, n_cls, 16, 16]
        orig_cams = F.conv2d(feature, self.cls_orig.conv.weight)  # [b, n_cls, 16, 16]
        orig_cams = F.relu(orig_cams)
        y = self.maxpool(y)
        return y.view(y.size(0), -1), orig_cams

    def cls_head_layer(self, feature):
        y = self.cls_layer(feature)
        layer_cams = F.conv2d(feature, self.cls_layer.conv.weight)
        layer_cams = F.relu(layer_cams)
        y = self.maxpool(y)
        return y.view(y.size(0), -1), layer_cams

    def encoder(self, x, is_layer):
        if is_layer:
            x1 = self.inc_layer(x)
            x2 = self.down1_layer(x1)
            x3 = self.down2_layer(x2)
            x4 = self.down3_layer(x3)
            x5 = self.down4_layer(x4)
        else:
            x1 = self.inc_orig(x)
            x2 = self.down1_orig(x1)
            x3 = self.down2_orig(x2)
            x4 = self.down3_orig(x3)
            x5 = self.down4_orig(x4)

        return [x1, x2, x3, x4, x5]

    def attn_encoder(self, org, layer):
        org1 = self.inc_orig(org)
        org2 = self.down1_orig(org1)
        org3 = self.down2_orig(org2)
        org4 = self.down3_orig(org3)
        layer1 = self.inc_layer(layer)
        layer2 = self.down1_layer(layer1)
        layer3 = self.down2_layer(layer2)
        layer4 = self.down3_layer(layer3)

        H, W = org4.size(2), org4.size(3)

        org4_reshape = org4.view(org4.size(0), org4.size(1), -1).permute(0, 2, 1)
        layer4_reshape = layer4.view(layer4.size(0), layer4.size(1), -1).permute(
            0, 2, 1
        )
        comb4 = torch.cat((org4_reshape, layer4_reshape), dim=1)
        comb4_f, _attn = self.attn1(comb4, H, W)
        org4 = (
            comb4_f[:, : int(comb4_f.size(1) / 2), :]
            .permute(0, 2, 1)
            .view(org4.size(0), org4.size(1), org4.size(2), org4.size(3))
        )
        layer4 = (
            comb4_f[:, int(comb4_f.size(1) / 2) :, :]
            .permute(0, 2, 1)
            .view(layer4.size(0), layer4.size(1), layer4.size(2), layer4.size(3))
        )

        org5 = self.down4_orig(org4)
        layer5 = self.down4_layer(layer4)

        H2, W2 = org5.size(2), org5.size(3)

        org5_reshape = org5.view(org5.size(0), org5.size(1), -1).permute(0, 2, 1)
        layer5_reshape = layer5.view(layer5.size(0), layer5.size(1), -1).permute(
            0, 2, 1
        )
        comb5 = torch.cat((org5_reshape, layer5_reshape), dim=1)
        comb5_f, _attn = self.attn2(comb5, H2, W2)
        org5 = (
            comb5_f[:, : int(comb5_f.size(1) / 2), :]
            .permute(0, 2, 1)
            .view(org5.size(0), org5.size(1), org5.size(2), org5.size(3))
        )
        layer5 = (
            comb5_f[:, int(comb5_f.size(1) / 2) :, :]
            .permute(0, 2, 1)
            .view(layer5.size(0), layer5.size(1), layer5.size(2), layer5.size(3))
        )

        return [org1, org2, org3, org4, org5], [layer1, layer2, layer3, layer4, layer5]

    # def decoder(self, comb_f_out):
    #     x1, x2, x3, x4, x5 = comb_f_out

    #     up_y = self.up1(x5, x4)
    #     up_y = self.up2(up_y, x3)
    #     up_y = self.up3(up_y, x2)
    #     up_y = self.up4(up_y, x1)
    #     up_y = self.outc(up_y)
    #     return up_y

    def get_cam_target_layers(self, type):
        if type == "x":
            cams = [self.down4_orig.maxpool_conv[-1]]
        elif type == "y":
            cams = [self.down4_layer]
        return cams

    def forward(self, input_data):
        orig_input = input_data[:, :3, :, :]
        layer_input = input_data[:, 3:6, :, :]
        orig_f_out, layer_f_out = self.attn_encoder(orig_input, layer_input)

        orig_feature = orig_f_out[-1]
        layer_feature = layer_f_out[-1]

        orig_cls_res, orig_cams = self.cls_head_orig(orig_feature)
        layer_cls_res, layer_cams = self.cls_head_layer(layer_feature)

        comb_cam = {
            "final_cam": orig_cams[:, 1:],
            "layer_cam": layer_cams[:, 1:],
            "orig_cam": orig_cams[:, 1:],
        }

        return [orig_cls_res], [layer_cls_res], None, comb_cam, None
