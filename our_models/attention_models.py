import torch.nn as nn
import timm
import  torch
import torch.nn.functional as F

DEIT_model_type = "deit_tiny_patch16_224"
SWIMT_model_type = 'swin_base_patch4_window7_224'

DIM_CHANGE_DEIT = {
    "deit_base_patch16_224": 768,
    "deit_tiny_patch16_224": 192
}

DIM_CHANGE_SWIMT = {
    "swin_base_patch4_window7_224": 128
}

class VitBlock(nn.Module):
    def __init__(self, num_class, num_input_channel=3):
        super(VitBlock, self).__init__()
        self.vit_model = torch.hub.load(
            "facebookresearch/deit:main", DEIT_model_type, pretrained=True
        )
        self.vit_model.patch_embed.proj = nn.Conv2d(
            num_input_channel, DIM_CHANGE_DEIT[DEIT_model_type], kernel_size=(16, 16), stride=(16, 16)
        )
        self.vit_model.head = nn.Linear(
            in_features=DIM_CHANGE_DEIT[DEIT_model_type], out_features=num_class, bias=True
        )

    def forward(self, input_tensor):
        resized_tensor = F.interpolate(
            input_tensor, size=(224, 224), mode="bilinear", align_corners=False
        )
        return self.vit_model(resized_tensor)


class SwimTBlock(nn.Module):
    def __init__(self, num_class, num_input_channel=3):
        super(SwimTBlock, self).__init__()
        self.swimT_model = timm.create_model(SWIMT_model_type, pretrained=True)
        
        self.swimT_model.patch_embed.proj = nn.Conv2d(
            num_input_channel, DIM_CHANGE_SWIMT[SWIMT_model_type], kernel_size=(4, 4), stride=(4, 4)
        )
        self.swimT_model.head = nn.Linear(
            in_features=1024, out_features=num_class, bias=True
        )

    def forward(self, input_tensor):
        resized_tensor = F.interpolate(
            input_tensor, size=(224, 224), mode="bilinear", align_corners=False
        )
        return self.swimT_model(resized_tensor)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, padding=1
            ),  # , bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels, out_channels, kernel_size=3, padding=1
            ),  # , bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class AttentionBlock(nn.Module):
    def __init__(
        self, in_channels, kernel_size=7, num_heads=4, image_size=16, inference=False
    ) -> None:
        super(AttentionBlock, self).__init__()
        self.kernel_size = min(
            kernel_size, image_size
        )  # receptive field shouldn't be larger than input H/W
        self.num_heads = num_heads
        self.dk = self.dv = in_channels
        self.dkh = self.dk // self.num_heads
        self.dvh = self.dv // self.num_heads

        assert (
            self.dk % self.num_heads == 0
        ), "dk should be divided by num_heads. (example: dk: 32, num_heads: 8)"
        assert (
            self.dk % self.num_heads == 0
        ), "dv should be divided by num_heads. (example: dv: 32, num_heads: 8)"

        self.k_conv = nn.Conv2d(self.dk, self.dk, kernel_size=1)  # .to(device)
        self.q_conv = nn.Conv2d(self.dk, self.dk, kernel_size=1)  # .to(device)
        self.v_conv = nn.Conv2d(self.dv, self.dv, kernel_size=1)  # .to(device)

        # Positional encodings
        self.rel_encoding_h = nn.Parameter(
            torch.randn(self.dk // 2, self.kernel_size, 1), requires_grad=True
        )
        self.rel_encoding_w = nn.Parameter(
            torch.randn(self.dk // 2, 1, self.kernel_size), requires_grad=True
        )

        # self.softmax = nn.Softmax2d()
        self.relu = nn.ReLU(inplace=True)
        # self.gelu = nn.GELU()
        
        # later access attention weights
        self.inference = inference
        if self.inference:
            self.register_parameter("weights", None)

    def forward(self, input_x):  # batch, 3, 512, 512
        x_rescale = input_x#self.down_scale(input_x)

        batch_size, _, height, width = x_rescale.size()
        # Compute k, q, v
        padded_x = F.pad(
            x_rescale,
            [
                (self.kernel_size - 1) // 2,
                (self.kernel_size - 1) - ((self.kernel_size - 1) // 2),
                (self.kernel_size - 1) // 2,
                (self.kernel_size - 1) - ((self.kernel_size - 1) // 2),
            ],
        )
        k = self.k_conv(padded_x)
        q = self.q_conv(x_rescale)
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
        qk = qk.reshape(
            batch_size,
            self.num_heads,
            height,
            width,
            self.kernel_size,
            self.kernel_size,
        )
        # Add positional encoding
        qr_h = torch.einsum("bhxydz,cij->bhxyij", q, self.rel_encoding_h)
        qr_w = torch.einsum("bhxydz,cij->bhxyij", q, self.rel_encoding_w)
        qk += qr_h
        qk += qr_w

        qk = qk.reshape(
            batch_size,
            self.num_heads,
            height,
            width,
            1,
            self.kernel_size * self.kernel_size,
        )
        # probably weights dropout after softmax?
        weights = F.softmax(qk, dim=-1)

        if self.inference:
            self.weights = nn.Parameter(weights)

        attn_out = torch.matmul(weights, v.transpose(4, 5))
        attn_out = attn_out.reshape(batch_size, -1, height, width)
        # attn_out = self.softmax(attn_out)
        attn_out = self.relu(attn_out)
        # attn_out = self.gelu(attn_out)

        return attn_out

class EncoderBlock(nn.Module):
    def __init__(
        self, in_channels, num_heads=4, inference=False
    ):
        super(EncoderBlock, self).__init__()
        self.att_norm = nn.LayerNorm([in_channels, 16, 16])
        self.attn = AttentionBlock(in_channels=in_channels, num_heads=num_heads, inference=inference)
        self.ffn_norm = nn.LayerNorm([in_channels, 16, 16])
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, padding=1, stride=1),
            # nn.BatchNorm2d(in_channels * 2), # maybe remove the norm
            # nn.ReLU(inplace=True),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1, stride=1),
            nn.Dropout2d(0.1)
        )
    def forward(self, input_t):
        f = input_t
        norm_1 = self.att_norm(input_t)
        attn_out = self.attn(norm_1)
        add_on_1 = attn_out + f
        
        f = add_on_1
        norm_2 = self.ffn_norm(add_on_1)
        mlp_out = self.mlp(norm_2)
        f_output = mlp_out + f
        
        return f_output
        