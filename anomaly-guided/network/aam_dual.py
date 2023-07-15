import torch.nn as nn
import torchvision.models as models
import torch
from network.attention_models import AttentionBlock, EncoderBlock, DoubleConv, Down, VitBlock, SwimTBlock, CrossAttentionBlock
import torch.nn.functional as F
from network.unet_part import *
from network.utils import num_channels_fc


class DUAL_AAM(nn.Module):
    def __init__(
        self, backbone, num_class, num_input_channel=12, backbone_name="resnet18"
    ):
        super(DUAL_AAM, self).__init__()
        self.backbone_name = backbone_name
        self.enc1 = backbone  # take the model without classifier
        self.enc1.conv1 = nn.Conv2d(num_input_channel // 2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.enc1 = torch.nn.Sequential(*(list(self.enc1.children())[:-2])) # 1024
        self.down_scale = nn.Sequential(
            DoubleConv(3, 64), # 3
            Down(64, 128),
            Down(128, 256),
            Down(256, 512),
            Down(512, 1024),
            Down(1024, 2048),
        )
        self.att_model_1 = AttentionBlock(in_channels=num_channels_fc[backbone_name], num_heads=4)
        self.att_model_2 = AttentionBlock(in_channels=num_channels_fc[backbone_name], num_heads=4)

        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels_fc[backbone_name], num_class, kernel_size=1),
            nn.BatchNorm2d(num_class),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_channels_fc[backbone_name], num_class, kernel_size=1),
            nn.BatchNorm2d(num_class),
            nn.ReLU(inplace=True),
        )

        self.maxpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.cls_fc_1 = nn.Sequential(
            nn.Linear(num_class, 1024, bias=True),  # 1024, 512 for resent18
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            
            nn.Linear(1024, 512, bias=True),  # 1024, 512 for resent18
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(512, num_class, bias=True),
        )

        
    def get_cam_target_layers(self):
        res = [self.conv1, self.conv2] #  self.enc2[-1][-1]
        return res
    
    def forward(self, concated_tensors):
        # both are b, 6, w, h
        N, C, H, W = concated_tensors.shape
        # healthy_x = concated_tensors[:, 3:6]
        concat_x = concated_tensors[:,:6]
        enc_concat = self.enc1(concat_x) # b, 2048, 16, 16
        enc_concat = self.att_model_2(enc_concat) # b, 2048, 16, 16

        diff_x = concated_tensors[:, 6:]
        diff_rescale = self.down_scale(diff_x) # b, 2048, 16, 16
        att_out = self.att_model_1(diff_rescale) # b, 2048, 16, 16

        enc_concat = self.conv1(enc_concat) # b, cls, 16, 16
        att_out = self.conv2(att_out) # b, cls, 16, 16


        output_concat = self.maxpool(enc_concat * att_out) # b, cls, 1, 1
        prob_concat = output_concat.view(output_concat.size(0), -1) # b, cls
        cls_concat = self.cls_fc_1(prob_concat) # b, cls

        att_out= F.interpolate(
                att_out, # [n, cls, w, h]
                size=(H, W),
                mode="bilinear",
                align_corners=True,
            )
        
        return {
            'cls_con': cls_concat, # b, cls
            'att_con': att_out, # b, cls, 512, 512
        }


if __name__ == "__main__":
    RESNet = models.resnet50(pretrained=True)
    # print(RESNet)
    MultiModel = DUAL_AAM(
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

