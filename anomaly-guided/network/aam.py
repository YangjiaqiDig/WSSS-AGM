import torch.nn as nn
import torchvision.models as models
import torch
from network.attention_models import AttentionBlock, EncoderBlock, DoubleConv, Down, VitBlock, SwimTBlock, CrossAttentionBlock
from network.unet_part import *
from network.utils import num_channels_fc


class SegmentationBlock(nn.Module):
    def __init__(self, backbone_name, num_class):
        super(SegmentationBlock, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.up_1 = nn.ConvTranspose2d(
            in_channels=num_channels_fc[backbone_name],
            out_channels=256,
            kernel_size=2,
            stride=2,
        )
        self.up_2 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2
        )
        self.up_3 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2
        )
        self.up_4 = nn.ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=2, stride=2
        )
        self.up_5 = nn.ConvTranspose2d(
            in_channels=32, out_channels=16, kernel_size=2, stride=2
        )
        # Final output
        self.conv_final = nn.Conv2d(
            in_channels=16, out_channels=num_class, kernel_size=1, padding=0, stride=1
        )

    def forward(self, base_res):
        x = self.dropout(base_res)
        x = self.up_1(x)
        x = self.up_2(x)
        x = self.up_3(x)
        x = self.up_4(x)
        x = self.up_5(x)
        seg_out = self.conv_final(x)

        return seg_out

class ResTransBlock(nn.Module):
    def __init__(self, backbone_name, num_class):
        super(ResTransBlock, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(num_channels_fc[backbone_name], num_class, bias=True)
        self.vit = VitBlock(num_class, num_input_channel=6)
        # self.swimT = SwimTBlock(num_class)
        self.cls = nn.Linear(num_class * 2, num_class, bias=True)

    def forward(self, base_res, diff_tensor=None):
        max_x = self.avgpool(base_res)  # batch, 2048, 16, 16 -> batch, 2048, 1, 1
        max_x = max_x.view(max_x.size(0), -1)
        base_res = self.fc(max_x)
        transformer_enhance = base_res  
        if diff_tensor is not None:
            transformer_res = self.vit(diff_tensor)  # diff_tensor: batch, 3, 512, 512
            transformer_enhance = torch.cat((base_res, transformer_res), dim=1)

        pred_output = self.cls(transformer_enhance)
        return pred_output


class AAM(nn.Module):
    def __init__(
        self, backbone, num_class, num_input_channel=3, backbone_name="resnet18"
    ):
        super(AAM, self).__init__()
        self.backbone_name = backbone_name
        self.base_model = backbone  # take the model without classifier
        self.num_input_channel = num_input_channel

        self._change_input_channel()
        
        self.down_scale = nn.Sequential(
            DoubleConv(3, 64), # 3
            Down(64, 128),
            Down(128, 256),
            Down(256, 512),
            Down(512, 1024),
            Down(1024, num_channels_fc[self.backbone_name]),
        )
        self.att_model = AttentionBlock(in_channels=num_channels_fc[self.backbone_name], num_heads=4) #2048

        # self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1)) # can try max pooling instead of average pooling
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        # self.cls = nn.Linear(2048, num_class, bias=True) # 512 if not have enhance_conv, else 1024
        self.cls = nn.Sequential(
            nn.Linear(num_channels_fc[self.backbone_name], 1024, bias=True),  # 1024, 512 for resent18
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            
            nn.Linear(1024, 512, bias=True),  # 1024, 512 for resent18
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(512, num_class, bias=True),
        )
        
        self.SegNet = SegmentationBlock(backbone_name, num_class)
    
    def _change_input_channel(self):
        if 'resnet' in self.backbone_name:
            self.base_model.conv1 = nn.Conv2d(self.num_input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.base_model = torch.nn.Sequential(*(list(self.base_model.children())[:-2])) # 1024
        elif 'vgg' in self.backbone_name:
            self.base_model.features[0] = nn.Conv2d(self.num_input_channel, 64, kernel_size=3, stride=1, padding=1)
            self.base_model = torch.nn.Sequential(*(list(self.base_model.children())[:-2])) # 1024
        elif 'densenet' in self.backbone_name:
            self.base_model.features.conv0 = nn.Conv2d(self.num_input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.base_model = torch.nn.Sequential(*(list(self.base_model.children())[:-1])) # 1024
        elif 'mnasnet' in self.backbone_name:
            self.base_model.layers[0] = nn.Conv2d(self.num_input_channel, 32, kernel_size=3, stride=2, padding=1, bias=False)
            self.base_model = torch.nn.Sequential(*(list(self.base_model.children())[:-1])) # 1024
        else:
            raise NotImplementedError
        
    def get_cam_target_layers(self):
        res = [self.base_model[-1][-1], self.down_scale[-1]] # self.down_scale[-1], self.att_model
        # res = [self.enhance_conv, self.base_model[-1][-1], self.down_scale[-1]]
        # multi layer self-attentions:
        # res = [self.base_model[-1][-1], self.layer[-1]] # self.down_scale[-1]
        return res
    
    def forward(self, concated_tensors):
        concat_x = concated_tensors[:,:6]
        concat_out = self.base_model(concat_x) # b, 1024, 32, 32
        
        diff_x = concated_tensors[:, 6:]
        diff_rescale = self.down_scale(diff_x) # b, 1024, 32, 32
        att_out = self.att_model(diff_rescale) # b, 1024, 32, 32

        # combine out from cnn and self-att
        # combined_out = torch.cat((base_out, att_out), dim=1) # b, 2048, 16, 16
        combined_out = concat_out * att_out # b, 1024, 32, 32 # if use multiplication, then cannot use relu?
        
        # classification output, can remove the enhance
        # enhance_pred = self.enhance_conv(combined_out) # b, 512, 32, 32
        cls_preds = self.maxpool(combined_out) # b, c, 1, 1
        cls_preds = cls_preds.view(cls_preds.size(0), -1) # b, c

        cls_preds = self.cls(cls_preds)


        return cls_preds
    

if __name__ == "__main__":
    RESNet = models.resnet50(pretrained=True)
    # print(RESNet)
    MultiModel = AAM(
        RESNet, num_class=5, num_input_channel=12, backbone_name="resnet50"
    )
    print(MultiModel)
    print(MultiModel.get_cam_target_layers())

    # print(MultiModel.base_model[-1][-1])
    # print(MultiModel.ClassNet.att.down_scale[-1].maxpool_conv[-1])
    # print(MultiModel.ClassNet.vit.vit_model.blocks[-1].norm1)
    # print(MultiModel.ClassNet.swimT.swimT_model.layers[-1].blocks[-1].norm2)
    input_x = torch.rand(2, 6, 512, 512)
    input_diff = torch.rand(2, 6, 512, 512)
    concated_tensors = torch.cat((input_x, input_diff), dim=1)
    outputs = MultiModel(concated_tensors)
    import pdb; pdb.set_trace()

