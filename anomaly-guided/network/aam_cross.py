import torch.nn as nn
import torchvision.models as models
import torch
from network.attention_models import AttentionBlock, EncoderBlock, DoubleConv, Down, VitBlock, SwimTBlock, CrossAttentionBlock
import torch.nn.functional as F
from network.utils import num_channels_fc

class AAM_UPGRADE(nn.Module):
    def __init__(
        self, backbone, num_class, num_input_channel=12, backbone_name="resnet18"
    ):
        super(AAM_UPGRADE, self).__init__()
        self.backbone_name = backbone_name
        self.siamese_encoder = backbone  # take the model without classifier
        self.siamese_encoder.conv1 = nn.Conv2d(num_input_channel // 2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.siamese_encoder = torch.nn.Sequential(*(list(self.siamese_encoder.children())[:-2])) # 1024
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.cls_fc_1 = nn.Sequential(
            nn.Linear(num_channels_fc[backbone_name], 1024, bias=True),  # 1024, 512 for resent18
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            
            nn.Linear(1024, 512, bias=True),  # 1024, 512 for resent18
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),

            nn.Linear(512, num_class, bias=True),
        )

        self.cls_fc_2 = nn.Sequential(
            nn.Linear(num_channels_fc[backbone_name], 1024, bias=True),  # 1024, 512 for resent18
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            
            nn.Linear(1024, 512, bias=True),  # 1024, 512 for resent18
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),

            nn.Linear(512, num_class, bias=True),
        )

        self.att_model_1 = nn.Sequential(
            # CrossAttentionBlock(in_channels=2048, num_heads=4), #2048
            AttentionBlock(in_channels=2048, num_heads=4), #2048
            nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, num_class, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.BatchNorm2d(num_class),
            # nn.ReLU(inplace=True),            
        )

        self.att_model_2 = nn.Sequential(
            # CrossAttentionBlock(in_channels=2048, num_heads=4), #2048
            AttentionBlock(in_channels=2048, num_heads=4), #2048
            nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, num_class, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.BatchNorm2d(num_class),
            # nn.ReLU(inplace=True),            
        )

        self.cls_att = nn.Sequential(
            nn.Linear(num_class, 32, bias=True),  # 1024, 512 for resent18
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(32, num_class, bias=True),
        )

        
    def get_cam_target_layers(self):
        res = [self.siamese_encoder[-1][-1]] # self.down_scale[-1], self.att_model
        return res
    
    def forward(self, concated_tensors):
        # both are b, 6, w, h
        N, C, H, W = concated_tensors.shape
        # healthy_x = concated_tensors[:, 3:6]
        concat_x = concated_tensors[:,:6]
        diff_x = concated_tensors[:, 6:]

        enc_concat = self.siamese_encoder(concat_x) # b, 2048, 16, 16
        enc_diff = self.siamese_encoder(diff_x)

        # after relu as last layer
        # att_concat = self.att_model_1([enc_concat, enc_diff]) # b, cls, 16, 16   
        # att_diff = self.att_model_2([enc_diff, enc_concat]) # b, cls, 16, 16
        att_concat = self.att_model_1(enc_concat) # b, cls, 16, 16
        att_diff = self.att_model_2(enc_diff) # b, cls, 16, 16
        pool_att_concat = self.maxpool(att_concat) # b, cls, 1, 1
        pool_att_diff = self.maxpool(att_diff) # b, cls, 1, 1
        prob_att_concat = pool_att_concat.view(N, -1) # b, cls
        prob_att_diff = pool_att_diff.view(N, -1) # b, cls
        cls_att_concat = self.cls_att(prob_att_concat) # b, cls
        cls_att_diff = self.cls_att(prob_att_diff) # b, cls

        att_concat= F.interpolate(
                att_concat, # [n, cls, w, h]
                size=(H, W),
                mode="bilinear",
                align_corners=True,
            )
        att_diff= F.interpolate(
                att_diff, # [n, cls, w, h]
                size=(H, W),
                mode="bilinear",
                align_corners=True,
            )

        output_concat = self.maxpool(enc_concat) # b, 2048, 1, 1
        output_diff = self.maxpool(enc_diff) # b, 2048, 1, 1
        prob_concat = output_concat.view(output_concat.size(0), -1) # b, 2048
        prob_diff = output_diff.view(output_diff.size(0), -1) # b, 2048

        cls_concat = self.cls_fc_1(prob_concat) # b, cls
        cls_diff = self.cls_fc_2(prob_diff) # b, cls
        

        return {
            'cls_con': cls_concat, # b, cls
            'cls_diff': cls_diff, # b, cls
            'cls_att_con': cls_att_concat, # b, cls
            'cls_att_diff': cls_att_diff, # b, cls
            'att_con': att_concat, # b, cls, 512, 512
            'att_diff': att_diff, # b, cls, 512, 512
        }

if __name__ == "__main__":
    RESNet = models.resnet50(pretrained=True)
    # print(RESNet)
    MultiModel = AAM_UPGRADE(
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

