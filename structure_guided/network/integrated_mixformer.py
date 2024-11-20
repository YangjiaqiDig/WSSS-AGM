from network import org_mix_transformer
import torch.nn as nn
import torch
import torch.nn.functional as F
import pickle
from timm.models.layers import trunc_normal_


class AdaptiveLayer(nn.Module):
    def __init__(self, in_dim, n_ratio, out_dim):
        super().__init__()
        hidden_dim = int(in_dim * n_ratio)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class IntegratedMixformer(nn.Module):
    def __init__(
        self,
        layer_branch,
        clip_branch=False,
        backbone="mit_b2",
        cls_num_classes=3,
        stride=[4, 2, 2, 1],
        img_size=512,
        pretrained=True,
        pool_type="max",
        freeze_layers=None,
        clip_version="large",
    ):
        super().__init__()
        self.cls_num_classes = cls_num_classes
        self.cls_layer_classes = 2
        self.stride = stride
        self.img_size = img_size
        self.layer_branch = layer_branch
        self.clip_branch = clip_branch

        self.encoder_org = getattr(org_mix_transformer, backbone)(
            stride=self.stride, img_size=self.img_size
        )
        self.in_channels = self.encoder_org.embed_dims
        if layer_branch:
            self.encoder_layer = getattr(org_mix_transformer, backbone)(
                stride=self.stride, img_size=self.img_size
            )
            self.get_cross_attention_fuse()
            # self.get_concat_fuse()
            self.head_layer = nn.Conv2d(
                in_channels=self.in_channels[3],
                out_channels=self.cls_layer_classes,
                kernel_size=1,
                bias=False,
            )
        if not clip_branch or layer_branch:
            self.head_org = nn.Conv2d(
                in_channels=self.in_channels[3],  #
                out_channels=self.cls_num_classes,
                kernel_size=1,
                bias=False,
            )

        # self.comb_f = nn.Sequential(
        #     nn.Conv2d(self.in_channels[3] * 2, self.in_channels[3], kernel_size=1, bias=True),  # concat org and layer features
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.5, inplace=False),

        #     nn.Conv2d(self.in_channels[3], self.in_channels[3] // 2, kernel_size=1, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.5, inplace=False),
        # )
        # self.head_org = nn.Conv2d(self.in_channels[3] // 2, self.cls_num_classes, kernel_size=1, bias=True)

        if pool_type == "max":
            self.maxpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        else:
            self.maxpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        if pretrained:
            self.load_pretained(backbone)

        if freeze_layers is not None:
            self.freeze_front_layers(freeze_layers)

        if clip_branch:
            if clip_version == "base":
                clip_f_path = (
                    "/scr2/xhu/jiaqi/wsss/structure_guided/text_features/clip_f.pkl"
                )
                num_channels = 512
            elif clip_version == "large":
                clip_f_path = "/scr2/xhu/jiaqi/wsss/structure_guided/text_features/clip-vit-large-patch14-f.pkl"
                num_channels = 768
            with open(clip_f_path, "rb") as f:
                self.clip_f = pickle.load(f).cpu()
            self.clip_fc3 = AdaptiveLayer(num_channels, 0.5, self.in_channels[2])
            self.clip_fc4 = AdaptiveLayer(num_channels, 0.5, self.in_channels[3])
            self.logit_scale3 = nn.parameter.Parameter(torch.ones([1]) * 1 / 0.07)
            self.logit_scale4 = nn.parameter.Parameter(torch.ones([1]) * 1 / 0.07)

    def freeze_front_layers(self, start_layer="block2"):
        # freeze enoders project patch embedding
        print(f"===== Freezing layers before {start_layer} =====")
        is_freeze = True
        for name, param in self.encoder_org.named_parameters():
            if not is_freeze:
                continue
            if start_layer in name:
                is_freeze = False
            if param.requires_grad and is_freeze:
                param.requires_grad = False
            # what about the norm1.weight and norm1.bias?
        is_freeze = True
        if self.layer_branch:
            for name, param in self.encoder_layer.named_parameters():
                if not is_freeze:
                    continue
                if start_layer in name:
                    is_freeze = False
                if param.requires_grad and is_freeze:
                    param.requires_grad = False

    def get_concat_fuse(self):
        self.concat_fuse_stage2 = nn.Sequential(
            nn.Conv2d(
                self.in_channels[1] * 2, self.in_channels[1], kernel_size=1, bias=True
            ),  # concat org and layer features
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Conv2d(
                self.in_channels[1], self.in_channels[1] * 2, kernel_size=1, bias=True
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
        )
        self.concat_fuse_stage3 = nn.Sequential(
            nn.Conv2d(
                self.in_channels[2] * 2, self.in_channels[2], kernel_size=1, bias=True
            ),  # concat org and layer features
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Conv2d(
                self.in_channels[2], self.in_channels[2] * 2, kernel_size=1, bias=True
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
        )
        self.concat_fuse_stage4 = nn.Sequential(
            nn.Conv2d(
                self.in_channels[3] * 2, self.in_channels[3], kernel_size=1, bias=True
            ),  # concat org and layer features
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Conv2d(
                self.in_channels[3], self.in_channels[3] * 2, kernel_size=1, bias=True
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
        )

    def get_cross_attention_fuse(self):
        # self.cross_attn_stage1_org = getattr(org_mix_transformer, "Attention")(self.in_channels[0], num_heads=4, qkv_bias=True, sr_ratio=8)
        # self.cross_attn_stage1_layer = getattr(org_mix_transformer, "Attention")(self.in_channels[0], num_heads=4, qkv_bias=True, sr_ratio=8)
        self.cross_attn_stage2_org = getattr(org_mix_transformer, "Attention")(
            self.in_channels[1], num_heads=4, qkv_bias=True, sr_ratio=4
        )
        self.cross_attn_stage2_layer = getattr(org_mix_transformer, "Attention")(
            self.in_channels[1], num_heads=4, qkv_bias=True, sr_ratio=4
        )
        self.cross_attn_stage3_org = getattr(org_mix_transformer, "Attention")(
            self.in_channels[2], num_heads=4, qkv_bias=True, sr_ratio=2
        )
        self.cross_attn_stage3_layer = getattr(org_mix_transformer, "Attention")(
            self.in_channels[2], num_heads=4, qkv_bias=True, sr_ratio=2
        )
        # self.cross_attn_stage4_org = getattr(org_mix_transformer, "Attention")(self.in_channels[3], num_heads=4, qkv_bias=True, sr_ratio=1)
        # self.cross_attn_stage4_layer = getattr(org_mix_transformer, "Attention")(self.in_channels[3], num_heads=4, qkv_bias=True, sr_ratio=1)

    def load_pretained(self, backbone):
        # initialize org encoder
        print("===== Loading Pretrained Model =====")
        state_dict = torch.load("./pretrained/" + backbone + ".pth", map_location="cpu")
        state_dict.pop("head.weight")
        state_dict.pop("head.bias")
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if k in self.encoder_org.state_dict().keys()
        }
        self.encoder_org.load_state_dict(state_dict, strict=False)
        if self.layer_branch:
            self.encoder_layer.load_state_dict(state_dict, strict=False)

    def get_cam_target_layers(self, type):
        return [
            # self.encoder_org.norm2,
            self.encoder_org.norm3,
            self.encoder_org.norm4,
        ]

    def forward_both_stage1(self, org, layer):
        org, attns1_org = self.encoder_org.forward_stage1(org)  # 128
        layer, attns1_layer = self.encoder_layer.forward_stage1(layer)

        return org, layer

    def forward_both_stage2(self, org, layer):
        org, attns2_org = self.encoder_org.forward_stage2(org)
        layer, attns2_layer = self.encoder_layer.forward_stage2(layer)
        # clip
        # fused_fs = self.concat_fuse_stage2(torch.cat([org, layer], dim=1))
        # org_updated = org + fused_fs[:, :self.in_channels[1], :, :]
        # layer_updated = layer + fused_fs[:, self.in_channels[1]:, :, :]

        # cross attention between org and layer
        B, _, H, W = org.shape
        org = org.flatten(2).transpose(1, 2)
        layer = layer.flatten(2).transpose(1, 2)

        org_updated, att1 = self.cross_attn_stage2_org.forward_cross(layer, org, H, W)
        layer_updated, att2 = self.cross_attn_stage2_layer.forward_cross(
            org, layer, H, W
        )

        # skip connection
        org_updated = org_updated + org
        layer_updated = layer_updated + layer

        org_updated = org_updated.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        layer_updated = (
            layer_updated.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        )

        return org_updated, layer_updated

    def forward_both_stage3(self, org, layer):
        org, attns3_org = self.encoder_org.forward_stage3(org)
        layer, attns3_layer = self.encoder_layer.forward_stage3(layer)
        # clip
        # fused_fs = self.concat_fuse_stage3(torch.cat([org, layer], dim=1))
        # org_updated = org + fused_fs[:, :self.in_channels[2], :, :]
        # layer_updated = layer + fused_fs[:, self.in_channels[2]:, :, :]

        # cross attention between org and layer
        B, _, H, W = org.shape
        org = org.flatten(2).transpose(1, 2)
        layer = layer.flatten(2).transpose(1, 2)

        org_updated, att1 = self.cross_attn_stage3_org.forward_cross(layer, org, H, W)
        layer_updated, att2 = self.cross_attn_stage3_layer.forward_cross(
            org, layer, H, W
        )

        # skip connection
        org_updated2 = org_updated + org
        layer_updated2 = layer_updated + layer

        org_updated2 = org_updated2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        layer_updated2 = (
            layer_updated2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        )

        return org_updated2, layer_updated2, org_updated.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous(), layer_updated.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

    def forward_both_stage4(self, org, layer):
        org, attns4_org = self.encoder_org.forward_stage4(org)
        layer, attns4_layer = self.encoder_layer.forward_stage4(layer)
        # clip
        # fused_fs = self.concat_fuse_stage4(torch.cat([org, layer], dim=1))
        # org_updated = org + fused_fs[:, :self.in_channels[3], :, :]
        # layer_updated = layer + fused_fs[:, self.in_channels[3]:, :, :]

        # concat org and layer features
        # org = torch.cat([org, layer], dim=1)
        # org = self.comb_f(org)

        return org, layer

    def get_similarity_map(self, image_f, txt_f, logit_scale):
        image_feature = image_f.permute(0, 2, 3, 1).reshape(
            -1, image_f.shape[1]
        )  # b*16*16, 512
        image_feature = image_feature / image_feature.norm(
            dim=-1, keepdim=True
        )  # b*16*16, 512
        logits_per_image = (
            logit_scale * image_feature @ txt_f.T.float()
        )  # b*16*16, num_cls
        outs = logits_per_image.view(
            image_f.shape[0], image_f.shape[2], image_f.shape[3], -1
        ).permute(
            0, 3, 1, 2
        )  # b, num_cls, 16, 16
        orig_cams = outs.clone().detach()
        cls_res = self.maxpool(outs).view(-1, txt_f.shape[0])
        return orig_cams, cls_res

    def forward_org_only(self, input_data):
        org = input_data[:, :3, :, :]
        # layer = input_data[:, 3:6, :, :]
        org, attns = self.encoder_org(org)

        # clip features
        clip_txt_f = self.clip_f.to(org[-1].device)

        l_fea_4 = self.clip_fc4(clip_txt_f)  # num_cls, 512
        orig_cams_4, cls_res_4 = self.get_similarity_map(
            org[-1], l_fea_4, self.logit_scale4
        )

        l_fea_3 = self.clip_fc3(clip_txt_f)  # num_cls, 512
        orig_cams_3, cls_res_3 = self.get_similarity_map(
            org[-2], l_fea_3, self.logit_scale3
        )

        orig_cams_3 = orig_cams_3[:, 1:].detach() # remove background class
        orig_cams_4 = orig_cams_4[:, 1:].detach() # remove background class
        # orig_cams = F.conv2d(last_out, self.head_org.weight).detach()
        # cls_res = self.maxpool(last_out)
        # cls_res = self.head_org(cls_res)
        # cls_res = cls_res.view(cls_res.size(0), -1)

        cams_dict = {
            "clip-l3-sim": orig_cams_3,
            "clip-l4-sim": orig_cams_4,
            "clip-l3-relu-sim": F.relu(orig_cams_3),
            "clip-l4-relu-sim": F.relu(orig_cams_4),
        }
        cls_preds = [cls_res_4, cls_res_3]
        return cls_preds, None, cams_dict

    def forward_dual_clip_branches(self, input_data):
        org = input_data[:, :3, :, :]
        layer = input_data[:, 3:6, :, :]
        # clip features
        clip_txt_f = self.clip_f.to(org[-1].device)

        org, layer = self.forward_both_stage1(org, layer)
        org, layer = self.forward_both_stage2(org, layer)
        org, layer, prev_org, _ = self.forward_both_stage3(org, layer)
        # clip 3
        l_fea_3 = self.clip_fc3(clip_txt_f)  # num_cls, 512
        orig_cams_3, clip_cls_res_3 = self.get_similarity_map(
            prev_org, l_fea_3, self.logit_scale3
        )

        org, layer = self.forward_both_stage4(org, layer)
        # clip 4
        l_fea_4 = self.clip_fc4(clip_txt_f)  # num_cls, 512
        orig_cams_4, clip_cls_res_4 = self.get_similarity_map(
            org, l_fea_4, self.logit_scale4
        )

        orig_cams = F.conv2d(org, self.head_org.weight).detach()
        layer_cams = F.conv2d(layer, self.head_layer.weight).detach()

        orig_cams_3 = orig_cams_3[:, 1:].detach() # remove background class
        orig_cams_4 = orig_cams_4[:, 1:].detach() # remove background class
        orig_cams = orig_cams[:, 1:].detach() # remove background class

        cls_res = self.maxpool(org)
        cls_res = self.head_org(cls_res)
        cls_res = cls_res.view(cls_res.size(0), -1)

        cls_layer_res = self.maxpool(layer)
        cls_layer_res = self.head_layer(cls_layer_res)
        cls_layer_res = cls_layer_res.view(cls_layer_res.size(0), -1)

        cams_dict = {
            "main-cam": orig_cams,
            "main-relu-cam": F.relu(orig_cams),
            "clip-l3-sim": orig_cams_3,
            "clip-l4-sim": orig_cams_4,
            "clip-l3-relu-sim": F.relu(orig_cams_3),
            "clip-l4-relu-sim": F.relu(orig_cams_4),
        }

        cls_preds = [cls_res, clip_cls_res_3, clip_cls_res_4]

        return cls_preds, [cls_layer_res], cams_dict

    def forward_dual_branches(self, input_data):
        org = input_data[:, :3, :, :]
        layer = input_data[:, 3:6, :, :]
        # outs_org, outs_layer = [], []

        org, layer = self.forward_both_stage1(org, layer)
        org, layer = self.forward_both_stage2(org, layer)
        org, layer, _, _ = self.forward_both_stage3(org, layer)
        org, layer = self.forward_both_stage4(org, layer)

        orig_cams = F.conv2d(org, self.head_org.weight).detach()
        layer_cams = F.conv2d(layer, self.head_layer.weight).detach()

        cls_res = self.maxpool(org)
        cls_res = self.head_org(cls_res)
        cls_res = cls_res.view(cls_res.size(0), -1)

        # cls_layer_res = self.head_layer(layer)
        # # layer_cams = F.softmax(cls_layer_res, dim=1)
        # cls_layer_res = self.maxpool(cls_layer_res)

        cls_layer_res = self.maxpool(layer)
        cls_layer_res = self.head_layer(cls_layer_res)
        cls_layer_res = cls_layer_res.view(cls_layer_res.size(0), -1)

        comb_cams = F.relu(orig_cams[:, 1:])  # + F.relu(layer_cams[:, 1:] * 0.5)

        cams_dict = {
            "final_cam": comb_cams,
            "layer_cam": orig_cams[:, 2:],
            "orig_cam": orig_cams[:, 1:],
        }

        return [cls_res], [cls_layer_res], cams_dict

    def forward(self, input_data):
        if self.layer_branch:
            if self.clip_branch:
                return self.forward_dual_clip_branches(input_data)
            return self.forward_dual_branches(input_data)
        else:
            return self.forward_org_only(input_data)
