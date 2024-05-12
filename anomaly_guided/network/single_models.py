import torch.nn as nn
import torchvision.models as models
import torch

from network.attention_models import VitBlock, SwimTBlock

num_channels_fc = {"resnet18": 512, "resnet50": 2048}

class Transformers(nn.Module):
    def __init__(self, backbone, num_class, num_input_channel=3, backbone_name="resnet18"):
        super(Transformers, self).__init__()
        # self.vit = VitBlock(num_class, num_input_channel=num_input_channel)
        self.swimT = SwimTBlock(num_class, num_input_channel=num_input_channel)

    def get_cam_target_layers(self):
        # res = [self.vit.vit_model.blocks[-1].norm1]
        res = [self.swimT.swimT_model.layers[-1].blocks[-1].norm2]
        return res
    
    def forward(self, concated_tensors):
        nbr_channels_of_tensor = concated_tensors.shape[1]
        if nbr_channels_of_tensor == 9:
            input_tensor = concated_tensors[:, :6]  # 1st dim is batch
        else:
            input_tensor = concated_tensors
        # pred_output = self.vit(input_tensor)
        pred_output = self.swimT(input_tensor)

        return pred_output

class CNNs(nn.Module):
    def __init__(
        self, backbone, num_class, num_input_channel=3, backbone_name="resnet18"
    ):
        super(CNNs, self).__init__()
        self.backbone_name = backbone_name
        self.base_model = backbone  # take the model without classifier
        self.base_model.conv1 = nn.Conv2d(
            num_input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.base_model.fc = nn.Linear(num_channels_fc[backbone_name], num_class, bias=True)
        self.include_segment = False

    def get_cam_target_layers(self):
        res = [self.base_model.layer4[-1]]
        return res
    
    def forward(self, concated_tensors):
        nbr_channels_of_tensor = concated_tensors.shape[1]
        if nbr_channels_of_tensor == 9:
            input_tensor = concated_tensors[:, :6]  # 1st dim is batch
        else:
            input_tensor = concated_tensors

        # classification output
        multi_label_pred = self.base_model(input_tensor)

        return multi_label_pred

if __name__ == "__main__":
    RESNet = models.resnet50(pretrained=True)
    # print(RESNet)
    MultiModel = CNNs(
        RESNet, num_class=5, num_input_channel=6, backbone_name="resnet50"
    )
    # print(MultiModel)

    # print(MultiModel.base_model[-1][-1])
    # print(MultiModel.ClassNet.att.down_scale[-1].maxpool_conv[-1])
    print(MultiModel.base_model.layer4[-1])
    # print(MultiModel.ClassNet.swimT.swimT_model.layers[-1].blocks[-1].norm2)
    input_x = torch.rand(2, 6, 512, 512)
    input_diff = torch.rand(2, 3, 512, 512)
    concated_tensors = torch.cat((input_x, input_diff), dim=1)
    cls_pred, seg_pred = MultiModel(concated_tensors)
    print(cls_pred, cls_pred.shape)
    