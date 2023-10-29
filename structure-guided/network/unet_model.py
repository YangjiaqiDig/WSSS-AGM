import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def normalize_cam(cams_tensor):
    norm_cams = cams_tensor.detach().cpu().numpy() # [batch, class, h, w]
    cam_max = np.max(norm_cams, (2, 3), keepdims=True)
    cam_min = np.min(norm_cams, (2, 3), keepdims=True)
    # norm_cams[norm_cams < cam_min + 1e-5] = 0
    
    norm_cams = (norm_cams - cam_min) / (cam_max - cam_min + 1e-7)
    return torch.tensor(norm_cams).to(cams_tensor.device)
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
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
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
    def __init__(self, n_classes, n_classes_layer=2, n_channels=3, bilinear=False, is_size=(224, 224)):
        super(UNetWeakly, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_classes_layer = n_classes_layer
        self.bilinear = bilinear
        self.is_size = is_size
        factor = 2 if bilinear else 1
        self.feature_maps = {}

        self.inc_orig = (DoubleConv(n_channels, 64))
        self.down1_orig = (Down(64, 128))
        self.down2_orig = (Down(128, 256))
        self.down3_orig = (Down(256, 512))
        self.down4_orig = (Down(512, 1024 // factor))

        self.inc_layer = (DoubleConv(3, 64)) # might need change for layers (binarilized better?)
        self.down1_layer = (Down(64, 128))
        self.down2_layer = (Down(128, 256))
        self.down3_layer = (Down(256, 512))
        self.down4_layer = (Down(512, 1024 // factor))

        self.cls_orig = (OutConv(1024, n_classes))
        self.cls_layer = (OutConv(1024, n_classes_layer))

        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))

        # self.up1 = (Up(2048, 1024 // factor, bilinear))
        # self.up2 = (Up(1024, 512 // factor, bilinear))
        # self.up3 = (Up(512, 256 // factor, bilinear))
        # self.up4 = (Up(256, 128, bilinear))

        self.outc = (OutConv(64, n_classes))
        # self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
    
    def cls_head_orig(self, feature):
        y = self.cls_orig(feature) # x [b, 1024, 16, 16], y [b, n_cls, 16, 16]
        orig_cams = F.conv2d(feature, self.cls_orig.conv.weight) # [b, n_cls, 16, 16]
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
    
    def decoder(self, comb_f_out):
        x1, x2, x3, x4, x5 = comb_f_out

        up_y = self.up1(x5, x4)
        up_y = self.up2(up_y, x3)
        up_y = self.up3(up_y, x2)
        up_y = self.up4(up_y, x1)
        up_y = self.outc(up_y)
        return up_y

    def forward(self, input_data):
        orig_input = input_data[0]
        layer_input = input_data[1]
        orig_f_out = self.encoder(orig_input, is_layer=False)
        layer_f_out = self.encoder(layer_input, is_layer=True)

        orig_feature = orig_f_out[-1]
        layer_feature = layer_f_out[-1]

        orig_cls_res, orig_cams = self.cls_head_orig(orig_feature)
        layer_cls_res, layer_cams = self.cls_head_layer(layer_feature)
        comb_cam = self.uncertainty_map(orig_cams, layer_cams)

        comb_f_out = [o * l for o, l in zip(orig_f_out, layer_f_out)]  # element-wise addition
        # comb_f_out = [torch.cat((o, l), dim=1) for o, l in zip(orig_f_out, layer_f_out)]  # concatenate along channel axis
        seg_res = self.decoder(comb_f_out)
        seg_class_pred = self.maxpool(seg_res)
        seg_class_pred = seg_class_pred.view(seg_class_pred.size(0), -1)
        # seg_class_pred = None

        return orig_cls_res, layer_cls_res, seg_res, comb_cam, seg_class_pred
    
    def uncertainty_map(self, orig_CAM, layer_CAM):
        # orig_CAM: [b, n_classes, w, h]
        # layer_CAM: [b, 2, w, h]

        # Compute certainty for layer_CAM (take the second class, i.e., presence of a lesion)
        layer_prob = F.softmax(layer_CAM, dim=1)
        lesion_layer_certainty = layer_prob[:, 1:, :, :] # confidence of lesion, higher value lower certainty
        layer_certainty = lesion_layer_certainty.repeat(1, self.n_classes-1, 1, 1)

        # lesion_layer_certainty = normalize_cam(layer_CAM)
        # layer_certainty = lesion_layer_certainty[:, 1:, :, :].repeat(1, self.n_classes-1, 1, 1)
        # import pdb; pdb.set_trace()

        # Compute weighted CAMs
        weighted_CAMs = orig_CAM[:,1:] * layer_certainty # [b, n_classes-1, w, h], not include background-0
        rescaled_CAMs = F.interpolate(weighted_CAMs.detach(), size=self.is_size, mode='bilinear', align_corners=False)
        rescaled_layer_cam = F.interpolate(lesion_layer_certainty.detach(), size=self.is_size, mode='bilinear', align_corners=False)
        rescaled_orig_cam = F.interpolate(orig_CAM.detach()[:,1:], size=self.is_size, mode='bilinear', align_corners=False)

        return {"final_cam": rescaled_CAMs, "layer_cam": rescaled_layer_cam, "orig_cam": rescaled_orig_cam}
    

class UNetWeaklyBranch1(nn.Module):
    def __init__(self, n_classes, n_classes_layer=2, n_channels=3, bilinear=False, is_size=(224, 224)):
        super(UNetWeaklyBranch1, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_classes_layer = n_classes_layer
        self.bilinear = bilinear
        self.is_size = is_size
        factor = 2 if bilinear else 1
        self.feature_maps = {}

        self.inc_orig = (DoubleConv(n_channels, 64))
        self.down1_orig = (Down(64, 128))
        self.down2_orig = (Down(128, 256))
        self.down3_orig = (Down(256, 512))
        self.down4_orig = (Down(512, 1024 // factor))


        self.cls_orig = (OutConv(1024, n_classes))

        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))

        self.outc = (OutConv(64, n_classes))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # self.maxpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
    
    def cls_head_orig(self, feature):
        y = self.cls_orig(feature) # x [b, 1024, 16, 16], y [b, n_cls, 16, 16]
        orig_cams = F.conv2d(feature, self.cls_orig.conv.weight) # [b, n_cls, 16, 16]
        orig_cams = F.relu(orig_cams)
        y = self.avgpool(y)
        return y.view(y.size(0), -1), orig_cams

    def encoder(self, x):
        x1 = self.inc_orig(x)
        x2 = self.down1_orig(x1)
        x3 = self.down2_orig(x2)
        x4 = self.down3_orig(x3)
        x5 = self.down4_orig(x4)

        return [x1, x2, x3, x4, x5]
    
    def decoder(self, comb_f_out):
        x1, x2, x3, x4, x5 = comb_f_out

        up_y = self.up1(x5, x4)
        up_y = self.up2(up_y, x3)
        up_y = self.up3(up_y, x2)
        up_y = self.up4(up_y, x1)
        up_y = self.outc(up_y)
        return up_y

    def forward(self, input_data):
        orig_input = input_data[0]
        orig_f_out = self.encoder(orig_input)

        orig_feature = orig_f_out[-1]

        orig_cls_res, orig_cams = self.cls_head_orig(orig_feature)
        comb_cam = self.uncertainty_map(orig_cams)

        seg_res = self.decoder(orig_f_out)
        seg_class_pred = None

        return orig_cls_res, None, seg_res, comb_cam, seg_class_pred
    
    def uncertainty_map(self, orig_CAM):
        # orig_CAM: [b, n_classes, w, h]

        # Compute weighted CAMs
        rescaled_CAMs = F.interpolate(orig_CAM[:,1:].detach(), size=self.is_size, mode='bilinear', align_corners=False)
        # import pdb; pdb.set_trace()
        return {"final_cam": rescaled_CAMs, "layer_cam": rescaled_CAMs, "orig_cam": rescaled_CAMs}
    
class UNetWeaklyBranch2(nn.Module):
    def __init__(self, n_classes, n_classes_layer=2, n_channels=3, bilinear=False, is_size=(224, 224)):
        super(UNetWeaklyBranch2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_classes_layer = n_classes_layer
        self.bilinear = bilinear
        self.is_size = is_size
        factor = 2 if bilinear else 1
        self.feature_maps = {}

        self.inc_layer = (DoubleConv(3, 64)) # might need change for layers (binarilized better?)
        self.down1_layer = (Down(64, 128))
        self.down2_layer = (Down(128, 256))
        self.down3_layer = (Down(256, 512))
        self.down4_layer = (Down(512, 1024 // factor))

        self.cls_layer = (OutConv(1024, n_classes))

        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))

        self.outc = (OutConv(64, n_classes))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # self.maxpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

    def cls_head_layer(self, feature):
        y = self.cls_layer(feature)
        layer_cams = F.conv2d(feature, self.cls_layer.conv.weight)
        layer_cams = F.relu(layer_cams)
        y = self.avgpool(y)
        return y.view(y.size(0), -1), layer_cams
    
    def encoder(self, x):
        x1 = self.inc_layer(x)
        x2 = self.down1_layer(x1)
        x3 = self.down2_layer(x2)
        x4 = self.down3_layer(x3)
        x5 = self.down4_layer(x4)

        return [x1, x2, x3, x4, x5]
    
    def decoder(self, comb_f_out):
        x1, x2, x3, x4, x5 = comb_f_out

        up_y = self.up1(x5, x4)
        up_y = self.up2(up_y, x3)
        up_y = self.up3(up_y, x2)
        up_y = self.up4(up_y, x1)
        up_y = self.outc(up_y)
        return up_y

    def forward(self, input_data):
        layer_input = input_data[1]
        layer_f_out = self.encoder(layer_input)

        layer_feature = layer_f_out[-1]

        layer_cls_res, layer_cams = self.cls_head_layer(layer_feature)
        comb_cam = self.uncertainty_map(layer_cams)

        seg_res = self.decoder(layer_f_out)
        seg_class_pred = None

        return layer_cls_res, None, seg_res, comb_cam, seg_class_pred
    
    def uncertainty_map(self, layer_CAM):
        # layer_CAM: [b, 2, w, h]

        # Compute certainty for layer_CAM (take the second class, i.e., presence of a lesion)
        rescaled_CAMs = F.interpolate(layer_CAM[:,1:].detach(), size=self.is_size, mode='bilinear', align_corners=False)
        # import pdb; pdb.set_trace()
        return {"final_cam": rescaled_CAMs, "layer_cam": rescaled_CAMs, "orig_cam": rescaled_CAMs}
    
class UNetWeaklyBranch2Binary(nn.Module):
    def __init__(self, n_classes, n_classes_layer=2, n_channels=3, bilinear=False, is_size=(224, 224)):
        super(UNetWeaklyBranch2Binary, self).__init__()
        self.n_channels = n_channels
        self.n_classes_layer = n_classes_layer
        self.bilinear = bilinear
        self.is_size = is_size
        factor = 2 if bilinear else 1
        self.feature_maps = {}

        self.inc_layer = (DoubleConv(3, 64)) # might need change for layers (binarilized better?)
        self.down1_layer = (Down(64, 128))
        self.down2_layer = (Down(128, 256))
        self.down3_layer = (Down(256, 512))
        self.down4_layer = (Down(512, 1024 // factor))

        self.cls_layer = (OutConv(1024, n_classes_layer))

        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))

        self.outc = (OutConv(64, n_classes_layer))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # self.maxpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

    def cls_head_layer(self, feature):
        y = self.cls_layer(feature)
        layer_cams = F.conv2d(feature, self.cls_layer.conv.weight)
        layer_cams = F.relu(layer_cams)
        y = self.avgpool(y)
        return y.view(y.size(0), -1), layer_cams
    
    def encoder(self, x):
        x1 = self.inc_layer(x)
        x2 = self.down1_layer(x1)
        x3 = self.down2_layer(x2)
        x4 = self.down3_layer(x3)
        x5 = self.down4_layer(x4)

        return [x1, x2, x3, x4, x5]
    
    def decoder(self, comb_f_out):
        x1, x2, x3, x4, x5 = comb_f_out

        up_y = self.up1(x5, x4)
        up_y = self.up2(up_y, x3)
        up_y = self.up3(up_y, x2)
        up_y = self.up4(up_y, x1)
        up_y = self.outc(up_y)
        return up_y

    def forward(self, input_data):
        layer_input = input_data[1]
        layer_f_out = self.encoder(layer_input)

        layer_feature = layer_f_out[-1]

        layer_cls_res, layer_cams = self.cls_head_layer(layer_feature)
        comb_cam = self.uncertainty_map(layer_cams)

        seg_res = self.decoder(layer_f_out)
        seg_class_pred = None

        return None, layer_cls_res, seg_res, comb_cam, seg_class_pred
    
    def uncertainty_map(self, layer_CAM):
        # layer_CAM: [b, 2, w, h]

        # Compute certainty for layer_CAM (take the second class, i.e., presence of a lesion)
        rescaled_CAMs = F.interpolate(layer_CAM[:,1:].detach(), size=self.is_size, mode='bilinear', align_corners=False)
        # import pdb; pdb.set_trace()
        return {"final_cam": rescaled_CAMs, "layer_cam": rescaled_CAMs, "orig_cam": rescaled_CAMs}
  
if __name__ == "__main__":
    model = UNetWeakly(n_classes=3, n_channels=3, bilinear=False)
    # Generate random inputs
    batch_size = 4
    height, width = 224, 224
    orig_input = torch.randn(batch_size, 3, height, width)
    layer_input = torch.randn(batch_size, 3, height, width)
    input_data = [orig_input, layer_input]

    # Forward pass through the model
    orig_cls_res, layer_cls_res, seg_res, cams, seg_class_pred = model(input_data)

    # Print output shapes
    print("Orig Cls Result Shape: ", orig_cls_res.shape)
    print("Layer Cls Result Shape: ", layer_cls_res.shape)
    if seg_class_pred is not None:
        print("Seg Class Pred Shape: ", seg_class_pred.shape)
    print("Segmentation Result Shape: ", seg_res.shape)
    print("CAMs Shape: ", cams["final_cam"].shape, cams["layer_cam"].shape, cams["orig_cam"].shape)
