from this import d
from turtle import forward, pd
from xml.etree.ElementInclude import include
import torch.nn as nn
import torchvision.models as models
import torch
import torch.nn.functional as F


num_channels_fc = {
    'resnet18': 512,
    'resnet50': 2048
}

class BaseModel(nn.Module):
    def __init__(self, backbone, num_input_channel):
        super(BaseModel, self).__init__()
        self.base_model = backbone  # take the model without classifier
        self.base_model.conv1 = nn.Conv2d(num_input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model = torch.nn.Sequential(*(list(self.base_model.children())[:-2]))
    def forward(self, x):
        y = self.base_model(x)
        return y

class SegmentationBlock(nn.Module):
    def __init__(self, backbone_name, num_class):
        super(SegmentationBlock, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.up_1 = nn.ConvTranspose2d(in_channels=num_channels_fc[backbone_name], out_channels=256, kernel_size=2, stride=2)
        self.up_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2) 
        self.up_4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.up_5 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2)
        # Final output
        self.conv_final = nn.Conv2d(in_channels=16, out_channels=num_class,
                                    kernel_size=1, padding=0, stride=1)
        
    def forward(self, base_res):
        x = self.dropout(base_res)
        x =  self.up_1(x)
        x = self.up_2(x)
        x = self.up_3(x)
        x = self.up_4(x)
        x = self.up_5(x)
        seg_out = self.conv_final(x)

        return seg_out

class AttentionBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=7, num_heads=4, image_size=512, inference=False) -> None:
        super(AttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(3, in_channels, kernel_size=3, padding=1, stride=1)
        
        self.kernel_size = min(kernel_size, image_size) # receptive field shouldn't be larger than input H/W         
        self.num_heads = num_heads
        self.dk = self.dv = in_channels
        self.dkh = self.dk // self.num_heads
        self.dvh = self.dv // self.num_heads

        assert self.dk % self.num_heads == 0, "dk should be divided by num_heads. (example: dk: 32, num_heads: 8)"
        assert self.dk % self.num_heads == 0, "dv should be divided by num_heads. (example: dv: 32, num_heads: 8)"  
        
        self.k_conv = nn.Conv2d(self.dk, self.dk, kernel_size=1)#.to(device)
        self.q_conv = nn.Conv2d(self.dk, self.dk, kernel_size=1)#.to(device)
        self.v_conv = nn.Conv2d(self.dv, self.dv, kernel_size=1)#.to(device)
        
        # Positional encodings
        self.rel_encoding_h = nn.Parameter(torch.randn(self.dk // 2, self.kernel_size, 1), requires_grad=True)
        self.rel_encoding_w = nn.Parameter(torch.randn(self.dk // 2, 1, self.kernel_size), requires_grad=True)
        
        # later access attention weights
        self.inference = inference
        if self.inference:
            self.register_parameter('weights', None)
        
    def forward(self, input_x):
        batch_size, _, height, width = input_x.size()
        x = self.conv1(input_x)
        # Compute k, q, v
        padded_x = F.pad(x, [(self.kernel_size-1)//2, (self.kernel_size-1)-((self.kernel_size-1)//2), (self.kernel_size-1)//2, (self.kernel_size-1)-((self.kernel_size-1)//2)])
        k = self.k_conv(padded_x)
        q = self.q_conv(x)
        v = self.v_conv(padded_x)
        # Unfold patches into [BS, num_heads*depth, horizontal_patches, vertical_patches, kernel_size, kernel_size]
        k = k.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
        v = v.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)

        # Reshape into [BS, num_heads, horizontal_patches, vertical_patches, depth_per_head, kernel_size*kernel_size]
        k = k.reshape(batch_size, self.num_heads, height, width, self.dkh, -1)
        v = v.reshape(batch_size, self.num_heads, height, width, self.dvh, -1)
        
        # Reshape into [BS, num_heads, height, width, depth_per_head, 1]
        q = q.reshape(batch_size, self.num_heads, height, width, self.dkh, 1)

        qk = torch.matmul(q.transpose(4, 5), k)    
        qk = qk.reshape(batch_size, self.num_heads, height, width, self.kernel_size, self.kernel_size)
        # Add positional encoding
        qr_h = torch.einsum('bhxydz,cij->bhxyij', q, self.rel_encoding_h)
        qr_w = torch.einsum('bhxydz,cij->bhxyij', q, self.rel_encoding_w)
        qk += qr_h
        qk += qr_w
        
        qk = qk.reshape(batch_size, self.num_heads, height, width, 1, self.kernel_size*self.kernel_size)
        weights = F.softmax(qk, dim=-1)    
        
        if self.inference:
            self.weights = nn.Parameter(weights)
        
        attn_out = torch.matmul(weights, v.transpose(4, 5)) 
        attn_out = attn_out.reshape(batch_size, -1, height, width)
        return attn_out
    

class ResNetBlock(nn.Module):
    def __init__(self, backbone_name, num_class):
        super(ResNetBlock, self).__init__()  
        self.att = AttentionBlock(32)
        self.maxpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))#nn.MaxPool2d(kernel_size=7, stride=7, padding=0)
        self.fc = nn.Linear(num_channels_fc[backbone_name], num_class) # 512 fore resent18
        self.sigmoid = nn.Sigmoid()  
        self.conv_expand = nn.Conv2d(32, num_channels_fc[backbone_name], kernel_size=3, padding=1, stride=1)

    def forward(self, base_res, diff_tensor=None):
        att_enhance = base_res
        if diff_tensor is not None:
            att_res = self.att(diff_tensor)
            att_scale = F.interpolate(att_res, size=base_res.shape[-2:], mode='bilinear', align_corners=False)
            att_enhance = base_res + self.conv_expand(att_scale)
        
        max_x = self.maxpool(att_enhance)
        max_x = max_x.view(max_x.size(0), -1)
        fc_x = self.fc(max_x)
        return fc_x

class MultiTaskModel(nn.Module):
    def __init__(self, backbone, num_class, num_input_channel=3, backbone_name='resnet18'):
        super(MultiTaskModel, self).__init__()
        self.backbone_name = backbone_name
        self.SharedNet = BaseModel(backbone, num_input_channel)
        self.ClassNet = ResNetBlock(backbone_name, num_class)
        self.SegNet = SegmentationBlock(backbone_name, num_class)
        self.cls_only = False
        self.include_segment = False
    
    def assign_conditions(self, cls_only_key=False, include_seg_key=False):
        self.cls_only = cls_only_key
        self.include_segment = include_seg_key
    
    def forward(self, concated_tensors):
        nbr_channels_of_tensor = concated_tensors.shape[1]
        if nbr_channels_of_tensor == 9:
            input_tensor = concated_tensors[:, :6] # 1st dim is batch
            diff_tensor = concated_tensors[:, 6:]
        else:
            input_tensor = concated_tensors
            diff_tensor = None
            
        base_output = self.SharedNet(input_tensor)
        # classification output
        multi_label_pred = self.ClassNet(base_output, diff_tensor)
        if self.cls_only:
            return multi_label_pred
        
        segmentation_pred = None
        # segmentation output
        if self.include_segment:
            segmentation_pred = self.SegNet(base_output)
            
        return multi_label_pred, segmentation_pred
    

if __name__ == "__main__":
    RESNet = models.resnet50(pretrained=True)
    MultiModel = MultiTaskModel(RESNet, num_class=5, num_input_channel=3, backbone_name='resnet50')
    print(MultiModel)
    input_x = torch.rand(2, 3, 512, 512)
    input_diff = torch.rand(2, 3, 512, 512)
    cls_pred, seg_pred = MultiModel(input_x, include_segment=True, diff_tensor=input_diff)
    print(cls_pred.shape, seg_pred.shape)


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