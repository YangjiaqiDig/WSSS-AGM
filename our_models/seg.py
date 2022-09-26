import torch.nn as nn

num_channels_fc = {"resnet18": 512, "resnet50": 2048}

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
