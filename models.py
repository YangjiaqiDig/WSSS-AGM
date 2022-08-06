from turtle import forward
import torch.nn as nn
import torchvision.models as models
import torch

num_channels_fc = {
    'resnet18': 512,
    'resnet50': 2048
}

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
        nn.ReLU(inplace=True),
    )

class U_Net(nn.Module):
    def __init__(self, multi_task_model, num_class, backbone_name='resnet18'):
        super(U_Net, self).__init__()
        self.multi_task_model = multi_task_model
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(num_channels_fc[backbone_name], 512 * 512)
        self.dropout = nn.Dropout(0.1)
        
        self.conv1_block = double_conv(1, 32)
        self.conv2_block = double_conv(32, 64)
        self.conv3_block = double_conv(64, 128)
        self.conv4_block = double_conv(128, 256)
        self.conv5_block = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.up_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.conv_up_1 = double_conv(512, 256)
        self.up_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.conv_up_2 = double_conv(256, 128)
        self.up_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2) 
        self.conv_up_3 = double_conv(128, 64)
        self.up_4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.conv_up_4 = double_conv(64, 32)
        # Final output
        self.conv_final = nn.Conv2d(in_channels=32, out_channels=num_class,
                                    kernel_size=1, padding=0, stride=1)
        
        self.softmax = nn.Softmax2d()

        
    def forward(self, x):
        x = self.multi_task_model(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.fc(x))
        x = x.view(x.size(0), 1, 512, 512)
                
        conv1 = self.conv1_block(x)
        x = self.maxpool(conv1)
        conv2 = self.conv2_block(x)
        x = self.maxpool(conv2)
        conv3 = self.conv3_block(x)
        x = self.maxpool(conv3)
        conv4 = self.conv4_block(x)
        x = self.maxpool(conv4)
        x = self.conv5_block(x)
        x =  self.up_1(x)
        x = torch.cat([x, conv4], dim=1)
        x = self.conv_up_1(x)
        x = self.up_2(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.conv_up_2(x)
        x = self.up_3(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.conv_up_3(x)
        x = self.up_4(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.conv_up_4(x)
        out = self.conv_final(x)
        # out = self.softmax(out)

        return out


class Up_Sample(nn.Module):
    def __init__(self, multi_task_model, num_class, backbone_name='resnet18'):
        super(Up_Sample, self).__init__()
        self.multi_task_model = multi_task_model
        self.dropout = nn.Dropout(0.1)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.up_1 = nn.ConvTranspose2d(in_channels=num_channels_fc[backbone_name], out_channels=256, kernel_size=2, stride=2)
        self.conv_up_1 = double_conv(512, 256)
        self.up_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.conv_up_2 = double_conv(256, 128)
        self.up_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2) 
        self.conv_up_3 = double_conv(128, 64)
        self.up_4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.conv_up_4 = double_conv(64, 32)
        self.up_5 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.conv_up_5 = double_conv(32, 16)
        # Final output
        self.conv_final = nn.Conv2d(in_channels=16, out_channels=num_class,
                                    kernel_size=1, padding=0, stride=1)
        
        
    def forward(self, x):
        x = self.multi_task_model(x)
        x = self.dropout(x)
        x =  self.up_1(x)
        x = self.up_2(x)
        x = self.up_3(x)

        x = self.up_4(x)

        x = self.up_5(x)
        out = self.conv_final(x)

        return out

class CAM_Net(nn.Module):
    def __init__(self, multi_task_model, num_class, backbone_name='resnet18'):
        super().__init__()
        self.multi_task_model = multi_task_model
        self.maxpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))#nn.MaxPool2d(kernel_size=7, stride=7, padding=0)
        self.fc = nn.Linear(num_channels_fc[backbone_name], num_class) # 512 fore resent18
        self.sigmoid = nn.Sigmoid()  

    def forward(self, x):
        x = self.multi_task_model(x)
        # print(x.shape) # b, 2048,16, 16
        max_x = self.maxpool(x)
        # print(max_x.shape) # b, 2048, 1, 1
        max_x = max_x.view(max_x.size(0), -1)
        fc_x = self.fc(max_x)
        return fc_x

class Att_Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x):
        return

class MultiTaskModel(nn.Module):
    def __init__(self, backbone, num_input_channel=3):
        super().__init__()
        self.base_model = backbone  # take the model without classifier
        self.base_model.conv1 = nn.Conv2d(num_input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model = torch.nn.Sequential(*(list(self.base_model.children())[:-2]))

    def forward(self, x):
        y = self.base_model(x)
        return y

if __name__ == "__main__":
    RESNet = models.resnet50(pretrained=True)
    print(RESNet)
    shared_model = MultiTaskModel(RESNet)
    print(shared_model)
    input_x = torch.rand(2, 3, 512, 512)
    cam_model = CAM_Net(shared_model, 5, 'resnet50')
    cam_output = cam_model(input_x)
    print(cam_output.shape)
    print(cam_output)
    # print(sum(cam_output))
    
    seg_model = Up_Sample(shared_model, 5, 'resnet50')
    seg_output = seg_model(input_x)
    print(seg_output.shape)
    
    # import torch
    # import torch.backends.cudnn as cudnn
    # import pdb

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