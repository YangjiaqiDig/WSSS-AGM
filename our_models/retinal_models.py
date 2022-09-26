import torch.nn as nn
import torchvision.models as models
import torch
import copy
from our_models.attention_models import AttentionBlock, EncoderBlock, DoubleConv, Down, VitBlock, SwimTBlock
from our_models.seg import SegmentationBlock

num_channels_fc = {"resnet18": 512, "resnet50": 2048}

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


class ResAttBlock(nn.Module):
    def __init__(self, backbone_name, num_class):
        super(ResAttBlock, self).__init__()
        self.att = AttentionBlock(num_channels_fc[backbone_name])
        # self.postatt = nn.Sequential(
        #     nn.Conv2d(num_channels_fc[backbone_name], 1024, kernel_size=3, padding=1, stride=1),
        #     # nn.BatchNorm2d(1024),
        #     nn.ReLU(inplace=True)
        # )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # nn.MaxPool2d(kernel_size=7, stride=7, padding=0)
        self.cls = nn.Sequential(
            nn.Linear(
                num_channels_fc[backbone_name], 512, bias=True
            ),  # 1024, 512 for resent18
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(512, num_class, bias=True),
        )

    def forward(self, base_res, diff_tensor=None):
        att_enhance = base_res  # batch, 2048, 16, 16
        if diff_tensor is not None:
            att_res = self.att(diff_tensor)  # diff_tensor: batch, 2048, 512, 512
            att_enhance = base_res * att_res  # + wrap as nn. function to extract layer
            # att_enhance = self.postatt(att_enhance)

        max_x = self.avgpool(att_enhance)  # batch, 2048, 1, 1
        max_x = max_x.view(max_x.size(0), -1)

        pred_output = self.cls(max_x)
        return pred_output
    
    
class MultiTaskModel_att(nn.Module):
    def __init__(
        self, backbone, num_class, num_input_channel=3, backbone_name="resnet18"
    ):
        super(MultiTaskModel_att, self).__init__()
        self.backbone_name = backbone_name
        self.base_model = backbone  # take the model without classifier
        self.base_model.conv1 = nn.Conv2d(num_input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model = torch.nn.Sequential(*(list(self.base_model.children())[:-2])) # 1024
        self.down_scale = nn.Sequential(
            DoubleConv(3, 64),
            Down(64, 128),
            Down(128, 256),
            Down(256, 512),
            Down(512, 1024),
            Down(1024, 2048),
        )
        self.att_model = AttentionBlock(in_channels=2048, num_heads=4) #2048
        # self.att_model = EncoderBlock(in_channels=2048, num_heads=4)
        # self.layer = nn.ModuleList()
        # for _ in range(2):
        #     layer = EncoderBlock(in_channels=2048, num_heads=4)
        #     self.layer.append(copy.deepcopy(layer))
            
        # self.enhance_conv = nn.Sequential(
        #     nn.Conv2d(1024, 512, kernel_size=3, padding=1, stride=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        # )
        # self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1)) # can try max pooling instead of average pooling
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        # self.cls = nn.Linear(2048, num_class, bias=True) # 512 if not have enhance_conv, else 1024
        self.cls = nn.Sequential(
            nn.Linear(num_channels_fc[backbone_name], 1024, bias=True),  # 1024, 512 for resent18
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            
            nn.Linear(1024, 512, bias=True),  # 1024, 512 for resent18
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(512, num_class, bias=True),
        )
        
        self.SegNet = SegmentationBlock(backbone_name, num_class)
        self.include_segment = False

    def assign_conditions(self, include_seg_key=False):
        self.include_segment = include_seg_key
        
    def get_cam_target_layers(self):
        res = [self.base_model[-1][-1], self.down_scale[-1]] # self.down_scale[-1], self.att_model
        # res = [self.enhance_conv, self.base_model[-1][-1], self.down_scale[-1]]
        # multi layer self-attentions:
        # res = [self.base_model[-1][-1], self.layer[-1]] # self.down_scale[-1]
        return res
    
    def forward(self, concated_tensors):
        input_x = concated_tensors[:,:6]
        base_out = self.base_model(input_x) # b, 1024, 32, 32
        
        diff_x = concated_tensors[:, 6:]
        diff_rescale = self.down_scale(diff_x) # b, 1024, 32, 32
        att_out = self.att_model(diff_rescale) # b, 1024, 32, 32
        
        # for layer_block in self.layer:
        #     diff_rescale = layer_block(diff_rescale)
        #     # add norm after the last att_out? or add relu/gelu instead
        # att_out = diff_rescale
        
        # combine out from cnn and self-att
        # combined_out = torch.cat((base_out, att_out), dim=1) # b, 2048, 32, 32
        combined_out = base_out * att_out # b, 1024, 32, 32 # if use multiplication, then cannot use relu?
        
        # classification output, can remove the enhance
        # enhance_pred = self.enhance_conv(combined_out) # b, 512, 32, 32
        cls_preds = self.maxpool(combined_out) # b, c, 1, 1
        cls_preds = cls_preds.view(cls_preds.size(0), -1) # b, c

        cls_preds = self.cls(cls_preds)

        segmentation_pred = None
        # segmentation output
        if self.include_segment:
            segmentation_pred = self.SegNet(base_out)

        return cls_preds, segmentation_pred


class MultiTaskModel_v2(nn.Module):
    def __init__(
        self, backbone, num_class, num_input_channel=3, backbone_name="resnet18"
    ):
        super(MultiTaskModel_v2, self).__init__()
        self.backbone_name = backbone_name
        self.base_model = backbone  # take the model without classifier
        self.base_model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model = torch.nn.Sequential(*(list(self.base_model.children())[:-3])) 
        self.down_scale = nn.Sequential(
            DoubleConv(3, 64),
            Down(64, 128),
            Down(128, 256),
            Down(256, 512),
            Down(512, 1024),
            # Down(1024, 2048),
        )
        # self.ClassNet = VitBlock(num_class, 2048) #
        self.ClassNet = AttentionBlock(in_channels=2048)
        
        self.post_cls = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.enhance_conv = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1)) # can try max pooling instead of average pooling
        self.cls = nn.Linear(512, num_class, bias=True) # 1024 if not have enhance_conv
        
        self.SegNet = SegmentationBlock(backbone_name, num_class)
        self.include_segment = False

    def assign_conditions(self, include_seg_key=False):
        self.include_segment = include_seg_key

    def forward(self, concated_tensors):
        input_x = concated_tensors[:,:3]
        gan_x = concated_tensors[:, 3:6]
        base_out_1 = self.base_model(input_x) # b, 1024, 32, 32
        base_out_2 = self.base_model(gan_x) # b, 1024, 32, 32
        base_out_diff = torch.abs(base_out_1 - base_out_2)
        
        diff_x = concated_tensors[:, 6:]
        diff_rescale = self.down_scale(diff_x) # b, 1024, 32, 32
        # diff_rescale = F.interpolate(diff_x, size=(32, 32), mode="bilinear", align_corners=False)
        
        # Siamese net outputs + diff_downscale
        combined_diff = torch.cat((base_out_diff, diff_rescale), dim=1) # b, 2048, 32, 32
        
        # classification output
        multi_label_pred = self.ClassNet(combined_diff) # b, 2048, 32, 32 # base_output 
        enhance_pred = self.post_cls(multi_label_pred)
    
        # extra enhance prepare down channels, can remove if not use enhace from prev resnet feature map
        enhance_pred = enhance_pred * base_out_diff # b, 1024, 32, 32
        enhance_pred = self.enhance_conv(enhance_pred) # b, 512, 32, 32
        # enhance_pred = multi_label_pred + base_out_diff
        
        cls_preds = self.avgpool(enhance_pred)
        cls_preds = cls_preds.view(cls_preds.size(0), -1)

        cls_preds = self.cls(cls_preds)

        segmentation_pred = None
        # segmentation output
        if self.include_segment:
            segmentation_pred = self.SegNet(base_out_1)

        return cls_preds, segmentation_pred


class MultiTaskModel(nn.Module):
    def __init__(
        self, backbone, num_class, num_input_channel=3, backbone_name="resnet18"
    ):
        super(MultiTaskModel, self).__init__()
        self.backbone_name = backbone_name
        self.base_model = backbone  # take the model without classifier
        self.base_model.conv1 = nn.Conv2d(
            num_input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.base_model = torch.nn.Sequential(*(list(self.base_model.children())[:-2]))
        self.ClassNet = ResTransBlock(backbone_name, num_class)
        self.SegNet = SegmentationBlock(backbone_name, num_class)
        self.include_segment = False

    def assign_conditions(self, include_seg_key=False):
        self.include_segment = include_seg_key

    def forward(self, concated_tensors):
        nbr_channels_of_tensor = concated_tensors.shape[1]
        if nbr_channels_of_tensor == 9:
            input_tensor = concated_tensors[:, :6]  # 1st dim is batch
            diff_tensor = concated_tensors[:, 6:]
        else:
            input_tensor = concated_tensors
            diff_tensor = None

        base_output = self.base_model(input_tensor)
        # classification output
        multi_label_pred = self.ClassNet(base_output, diff_tensor) # base_output

        segmentation_pred = None
        # segmentation output
        if self.include_segment:
            segmentation_pred = self.SegNet(base_output)

        return multi_label_pred, segmentation_pred


if __name__ == "__main__":
    RESNet = models.resnet50(pretrained=True)
    # print(RESNet)
    MultiModel = MultiTaskModel_att(
        RESNet, num_class=5, num_input_channel=6, backbone_name="resnet50"
    )
    print(MultiModel)
    print(MultiModel.get_cam_target_layers())

    # print(MultiModel.base_model[-1][-1])
    # print(MultiModel.ClassNet.att.down_scale[-1].maxpool_conv[-1])
    # print(MultiModel.ClassNet.vit.vit_model.blocks[-1].norm1)
    # print(MultiModel.ClassNet.swimT.swimT_model.layers[-1].blocks[-1].norm2)
    input_x = torch.rand(2, 6, 512, 512)
    input_diff = torch.rand(2, 3, 512, 512)
    concated_tensors = torch.cat((input_x, input_diff), dim=1)
    cls_pred, seg_pred = MultiModel(concated_tensors)
    print(cls_pred, cls_pred.shape)
    
    # from utils import SegmentationModelOutputWrapper
    # def reshape_transform(tensor, height=14, width=14):
    #     if tensor.shape[1] != 197:
    #         return tensor
    #     print('1', tensor.shape)
    #     result = tensor[:, 1:, :].reshape(tensor.size(0),
    #                                     height, width, tensor.size(2))
    #     print('2', result.shape)

    #     # Bring the channels to the first dimension,
    #     # like in CNNs.
    #     result = result.transpose(2, 3).transpose(1, 2)
    #     print('3', result.shape)
    #     return result
    # from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad, GuidedBackpropReLUModel
    # target_layers=[MultiModel.base_model[-1][-1], MultiModel.ClassNet.vit.vit_model.blocks[-1].norm1]
    # cam = GradCAM(model=SegmentationModelOutputWrapper(MultiModel.to('cuda')),
    #             target_layers=target_layers,
    #             use_cuda='cuda',
    #             reshape_transform=reshape_transform)
    # gray = cam(input_tensor=concated_tensors.to('cuda'),
    #         targets=None, 
    #         eigen_smooth=False,
    #         aug_smooth=False)
    # print(cls_pred.shape, seg_pred)
    # from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

    # input_vit = torch.rand(2, 6, 224, 224)
    # VitModel = VitBlock(num_class=5, num_input_channel=6)
    # res = VitModel(input_vit)
    # print(res)

    # seg_model.eval()
    # cudnn.benchmark = True
    # seg_model = seg_model.cuda()
    # cam_model.eval()
    # cudnn.benchmark = True
    # cam_model = cam_model.cuda()

    # with torch.no_grad():
    #     for i in range(100):
    #         img = torch.randn(4, 3, 512, 512).cuda()
    #         c_pred = seg_model(img).squeeze()
    #         d_pred = cam_model(img)
    #         print(torch.cuda.memory_reserved()/1024/1024)
    #         print(torch.cuda.max_memory_reserved()/1024/1024)
    #         torch.cuda.empty_cache()
    #         print(torch.cuda.memory_reserved()/1024/1024)
    #         print(torch.cuda.max_memory_reserved()/1024/1024)
    #         pdb.set_trace()

    # A full forward pass
    # im = torch.randn(8, 3, 256, 256)
    # model = U_Net()
    # likelihood_map = model(im)
    # print(likelihood_map.shape)
    # # print(x.shape)
    # del model
