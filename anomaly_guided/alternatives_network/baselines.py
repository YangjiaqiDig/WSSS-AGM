
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet2(nn.Module):
    def __init__(self, block, layers, num_classes=2, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.inplanes2 = 1024
        self.dilation2 = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer11 = self._make_layer(block, 64, 3,flag=True)
        self.layer22 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer33 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        
        self.layer1 = self._make_layer2(block, 64, layers[0])

        self.layer2 = self._make_layer2(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer2(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer2(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.ada_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.ada_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # self.fc2 = nn.Linear(2, 1)
        self.fc2 = nn.Linear(1024, num_classes)
        self.sigmoid=nn.Sigmoid()
        self.softmax=nn.Softmax(dim=1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                # elif isinstance(m, BasicBlock):
                #     nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False,flag=False):
        if flag:
            self.dilation2=self.dilation
            self.inplanes2=self.inplanes
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
    def _make_layer2(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation2 = self.dilation2
        if dilate:
            self.dilation2 *= stride
            stride = 1
        if stride != 1 or self.inplanes2 != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes2, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes2, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation2, norm_layer))
        self.inplanes2 = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes2, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
    
    def get_cam_target_layers(self, branch):
        if branch == 'x':
            return [self.layer33]
        return [self.relu, self.layer2, self.layer1]

    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x) # 1/2
        
        # feature propagation encoder?
        y = self.maxpool(x) # 1/4
        y_1 = self.layer1(y)
        y_2 = self.layer2(y_1) # 1/8
        y_3 = self.layer3(y_2) # 1/16
        y_4 = self.layer4(y_3) # 1/32
        y_f = self.ada_avgpool(y_4)
        y_f = y_f.reshape(y_f.size(0), -1)
        y_f = self.fc(y_f) # (batch, num_classes)
        # y = self.softmax(y)
        # y = self.sigmoid(y) # remove, use BCEWithLogitsLoss instead

        # classification ?
        x_1 = self.layer11(x) # 1/2
        x_2 = self.layer22(x_1) # 1/4
        x_3 = self.layer33(x_2) # 1/8
        x_f = self.ada_maxpool(x_3)#所以这里都是均值 #(batch, 1248,1,1)
        # x = torch.mean(x,(1)) #(batch, 1,1)
        # x = x.view(x.size(0))
        x_f = x_f.reshape(x_f.size(0), -1)
        x_f= self.fc2(x_f) # (batch, num_classes)
        # x = self.sigmoid(x) # (batch) # remove, use BCEWithLogitsLoss instead

        out = OrderedDict()
        out['x'] = x_f
        out['y'] = y_f
        
        return out

from gan_inference import load_gan_model

class AutoBioDetectModel(nn.Module):
    def __init__(self, pretrained, num_class):
        # load GANomaly Discriminator with pretrained weights
        super(AutoBioDetectModel, self).__init__()
        features_layers = list(pretrained.netd.children())[0]
        # import pdb; pdb.set_trace()
        self.cv1 = nn.Sequential(*features_layers[:2]) # 1 / 2  -> 256
        self.cv2 = nn.Sequential(*features_layers[2:5]) # 1 / 4 -> 128
        self.cv3 = nn.Sequential(*features_layers[5:8]) # 1 / 8 -> 64
        self.cv4 = nn.Sequential(*features_layers[8:11]) # 1 / 16 -> 32
        self.cv5 = nn.Sequential(*features_layers[11:14]) # 1 / 32 -> 16
        self.cv6 = nn.Sequential(*features_layers[14:17]) # 1 / 64 -> 8
        self.cv7 = nn.Sequential(*features_layers[17:]) # 1 / 128 -> 4
        # self.backbone = nn.Sequential(*features_layers)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024, bias=True),  # 1024, 512 for resent18
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            
            nn.Linear(1024, 512, bias=True),  # 1024, 512 for resent18
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(512, num_class, bias=True),
        )
        # self.classifier = nn.Conv2d(in_channels=2048, out_channels=num_class, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        # self.backbone.classifier = nn.Conv2d(2048, 1, kernel_size=(4, 4), stride=(1, 1), bias=False)

    def get_cam_target_layers(self):
        return [self.cv7.requires_grad_(), self.cv6.requires_grad_(), self.cv5.requires_grad_(), self.cv4.requires_grad_()]

    def forward(self, input_tensor):
        f_out_1 = self.cv1(input_tensor)
        f_out_2 = self.cv2(f_out_1)
        f_out_3 = self.cv3(f_out_2)
        f_out_4 = self.cv4(f_out_3)
        f_out_5 = self.cv5(f_out_4)
        f_out_6 = self.cv6(f_out_5)
        f_out = self.cv7(f_out_6)
        # import pdb; pdb.set_trace()

        # cam_1= F.conv2d(f_out, self.classifier.weight).detach()
        # cam_1 = self.relu(cam_1)
        # import pdb; pdb.set_trace()

        y = self.avgpool(f_out)
        y = y.view(y.size(0), -1)
        y = self.classifier(y)
        return y