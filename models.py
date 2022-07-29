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
        return self.sigmoid(fc_x)

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
    input_x = torch.rand(2, 3, 512, 512)
    cam_model = CAM_Net(shared_model, 5, 'resnet50')
    cam_output = cam_model(input_x)
    print(cam_output.shape)
    
    seg_model = U_Net(shared_model, 5, 'resnet50')
    seg_output = seg_model(input_x)
    print(seg_output.shape)
    
    
     # A full forward pass
    # im = torch.randn(8, 3, 256, 256)
    # model = U_Net()
    # likelihood_map = model(im)
    # print(likelihood_map.shape)
    # # print(x.shape)
    # del model