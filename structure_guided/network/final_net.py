from network import org_mix_transformer
import torch.nn as nn
import torch
import torch.nn.functional as F
import pickle
from timm.models.layers import trunc_normal_


class AdaptiveLayerStage3(nn.Module):
    def __init__(self, in_dim, n_ratio, out_dim):
        super().__init__()
        hidden_dim = int(in_dim * n_ratio)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
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
        x = self.relu(x)  # ignore all the negative outputs
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class AdaptiveLayerStage4(nn.Module):
    def __init__(self, in_dim, n_ratio, out_dim):
        super().__init__()
        hidden_dim = int(in_dim * n_ratio)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
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
        x = self.relu(x)  # ignore all the negative outputs
        # x = self.fc2(x)
        # x = self.relu(x)
        x = self.fc3(x)
        return x


class IntegratedMixformer(nn.Module):
    def __init__(
        self,
        layer_branch,
        clip_f=None,
        backbone="mit_b2",
        cls_num_classes=3,
        stride=[4, 2, 2, 1],
        img_size=512,
        pretrained=True,
        pool_type="max",
        freeze_layers=None,
        caption_version="blip_norm_clip_base_embed",
        constraint_loss=False,
        caption_branch=False,
    ):
        super().__init__()
        self.cls_num_classes = cls_num_classes
        self.cls_layer_classes = 2
        self.stride = stride
        self.img_size = img_size
        self.layer_branch = layer_branch
        self.clip_f = clip_f
        self.caption_branch = caption_branch

        self.encoder_org = getattr(org_mix_transformer, backbone)(
            stride=self.stride, img_size=self.img_size
        )
        self.in_channels = self.encoder_org.embed_dims
        self.constraint_loss = constraint_loss

        self.head_org = nn.Conv2d(
            in_channels=self.in_channels[3],  #
            out_channels=self.cls_num_classes,
            kernel_size=1,
            bias=False,
        )

        if layer_branch:
            self.encoder_layer = getattr(org_mix_transformer, backbone)(
                stride=self.stride, img_size=self.img_size
            )
            self.get_cross_attention_fuse()
            self.head_layer = nn.Conv2d(
                in_channels=self.in_channels[3],
                out_channels=self.cls_layer_classes,
                kernel_size=1,
                bias=False,
            )
            # linear
            # self.head_layer = nn.Linear(self.in_channels[3], self.cls_layer_classes)
        if pool_type == "max":
            self.maxpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        else:
            self.maxpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        if pretrained:
            self.load_pretained(backbone)

        if freeze_layers is not None:
            self.freeze_front_layers(freeze_layers)

        if self.clip_f is not None:
            num_channels = self.clip_f.shape[1]
            self.clip_fc3 = AdaptiveLayerStage4(
                num_channels, 1 / 2, self.in_channels[2]
            )
            self.clip_fc4 = AdaptiveLayerStage4(
                num_channels, 1 / 2, self.in_channels[3]
            )
            self.logit_scale3 = nn.parameter.Parameter(torch.ones([1]) * 1)  # / 0.07
            self.logit_scale4 = nn.parameter.Parameter(torch.ones([1]) * 1)  # / 0.07

        if self.caption_branch:
            concate_channels = 512
            if "clip_large" in caption_version:
                concate_channels = 768
            if "minilm" in caption_version:
                concate_channels = 384
            self.fuse_conv = nn.Conv2d(
                in_channels=self.in_channels[3] + concate_channels,  # 768
                out_channels=self.in_channels[3],
                kernel_size=1,
                bias=False,
            )

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
            # start_layer="block2"
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
        self.cross_attn_stage2_org = getattr(org_mix_transformer, "Block")(
            self.in_channels[1], num_heads=4, qkv_bias=True, sr_ratio=4
        )
        self.cross_attn_stage2_layer = getattr(org_mix_transformer, "Block")(
            self.in_channels[1], num_heads=4, qkv_bias=True, sr_ratio=4
        )
        self.cross_attn_stage3_org = getattr(org_mix_transformer, "Block")(
            self.in_channels[2], num_heads=4, qkv_bias=True, sr_ratio=2
        )
        self.cross_attn_stage3_layer = getattr(org_mix_transformer, "Block")(
            self.in_channels[2], num_heads=4, qkv_bias=True, sr_ratio=2
        )

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
        if type == "x":
            return [
                # self.encoder_org.norm2,
                self.encoder_org.norm3,
                self.encoder_org.norm4,
            ]
        return [
            # self.encoder_layer.norm2,
            self.encoder_layer.norm3,
            self.encoder_layer.norm4,
        ]

    def forward_both_stage1(self, org, layer):
        org, attns1_org = self.encoder_org.forward_stage1(org)  # 128
        layer, attns1_layer = self.encoder_layer.forward_stage1(layer)

        return org, layer

    def forward_both_stage2(self, org, layer):
        org, attns2_org = self.encoder_org.forward_stage2(org)
        layer, attns2_layer = self.encoder_layer.forward_stage2(layer)

        return org, layer

    def exchange_attention_stage3(self, org, layer):
        # cross attention between org and layer
        B, _, H, W = org.shape
        org = org.flatten(2).transpose(1, 2)
        layer = layer.flatten(2).transpose(1, 2)

        org_updated, att1 = self.cross_attn_stage2_org.forward_cross(layer, org, H, W)
        layer_updated, att2 = self.cross_attn_stage2_layer.forward_cross(
            org, layer, H, W
        )

        org_updated = org_updated.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        layer_updated = (
            layer_updated.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        )
        return org_updated, layer_updated

    def forward_both_stage3(self, org, layer):
        org_updated, layer_updated = self.exchange_attention_stage3(org, layer)
        org, attns3_org = self.encoder_org.forward_stage3(org_updated)
        layer, attns3_layer = self.encoder_layer.forward_stage3(layer_updated)

        return org, layer

    def exchange_attention_stage4(self, org, layer):
        # cross attention between org and layer
        B, _, H, W = org.shape
        org = org.flatten(2).transpose(1, 2)
        layer = layer.flatten(2).transpose(1, 2)

        org_updated, att1 = self.cross_attn_stage3_org.forward_cross(layer, org, H, W)
        layer_updated, att2 = self.cross_attn_stage3_layer.forward_cross(
            org, layer, H, W
        )
        org_updated = org_updated.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        layer_updated = (
            layer_updated.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        )
        return org_updated, layer_updated

    def forward_both_stage4(self, org, layer):
        org_updated, layer_updated = self.exchange_attention_stage4(org, layer)
        org, attns4_org = self.encoder_org.forward_stage4(org_updated)
        layer, attns4_layer = self.encoder_layer.forward_stage4(layer_updated)

        return org, layer

    def get_sim_stage3(self, image_f, txt_f, logit_scale):
        # txt_f: num_cls, 320
        image_feature = image_f.permute(0, 2, 3, 1).reshape(
            -1, image_f.shape[1]
        )  # b*32*32, 320
        image_feature = image_feature / image_feature.norm(
            dim=-1, keepdim=True
        )  # b*32*32, 320
        logits_per_image = (
            logit_scale * image_feature @ txt_f.T.float()
        )  # b*32*32, num_cls
        outs = logits_per_image.view(
            image_f.shape[0], image_f.shape[2], image_f.shape[3], -1
        ).permute(
            0, 3, 1, 2
        )  # b, num_cls, 32, 32
        # stage3_out is b, num_cls, 32, 32, current outs is b, num_cls, 16, 16
        # outs = F.interpolate(outs, size=(16, 16), mode="bilinear", align_corners=False)
        # outs = self.downscale_stage3(outs)

        # _outs = self.similarity_head3(outs)
        orig_cams = outs.clone()
        cls_res = self.maxpool(outs).view(-1, txt_f.shape[0])
        return orig_cams, cls_res

    def get_sim_stage4(self, image_f, txt_f, logit_scale):
        # txt_f: num_cls, 512
        image_feature = image_f.permute(0, 2, 3, 1).reshape(-1, image_f.shape[1])
        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
        logits_per_image = logit_scale * image_feature @ txt_f.T.float()
        outs = logits_per_image.view(
            image_f.shape[0], image_f.shape[2], image_f.shape[3], -1
        ).permute(0, 3, 1, 2)
        orig_cams = outs.clone()
        cls_res = self.maxpool(outs).view(-1, txt_f.shape[0])
        return orig_cams, cls_res

    def fuse_caption_feature(self, v_f, c_f):
        batch_size, _, H, W = v_f.shape
        c_f = c_f.view(batch_size, -1, 1, 1)  # Reshape to (batch, 512, 1, 1)
        c_f = c_f.expand(-1, -1, H, W)  # Expand to (batch, 512, 16, 16)

        # Combine the features
        # Option 1: Concatenate features
        fuse_f = torch.cat([v_f, c_f], dim=1)  # (batch, 1024, 16, 16)
        # Option 2: Element-wise addition (alternative to concatenation, no dimension change)
        # fuse_f = v_f + c_f
        fuse_f = self.fuse_conv(fuse_f)

        # fuse_f = F.relu(fuse_f)

        return fuse_f

    def forward_dual_clip_branches(self, input_data, caption_input=None):
        org = input_data[:, :3, :, :]
        layer = input_data[:, 3:6, :, :]
        # clip features
        clip_txt_f = self.clip_f.to(org.device)  # cls, 768

        org, layer = self.forward_both_stage1(org, layer)
        org, layer = self.forward_both_stage2(org, layer)
        org, layer = self.forward_both_stage3(org, layer)

        # clip 3
        l_fea_3 = self.clip_fc3(clip_txt_f)  # num_cls, 320
        visual_f = org + layer
        clip_cams_3, clip_cls_res_3 = self.get_sim_stage3(
            visual_f, l_fea_3, self.logit_scale3
        )  # clip_cams_3, clip_cls_res_3

        org, layer = self.forward_both_stage4(org, layer)
        if self.caption_branch:
            # caption_input: batch, 512
            caption_input = caption_input.to(org.device)
            org = self.fuse_caption_feature(org, caption_input)

        # clip 4
        l_fea_4 = self.clip_fc4(clip_txt_f)  # num_cls, 512
        visual_f = org + layer
        clip_cams_4, clip_cls_res_4 = self.get_sim_stage4(
            visual_f, l_fea_4, self.logit_scale4
        )

        orig_cams = F.conv2d(org, self.head_org.weight)
        layer_cams = F.conv2d(
            layer, self.head_layer.weight
        )  # .unsqueeze(-1).unsqueeze(-1))
        if not self.constraint_loss:
            orig_cams = orig_cams.detach()
            layer_cams = layer_cams.detach()
            clip_cams_3 = clip_cams_3.detach()
            clip_cams_4 = clip_cams_4.detach()

        cls_res = self.maxpool(org)
        cls_res = self.head_org(cls_res)
        cls_res = cls_res.view(cls_res.size(0), -1)

        cls_layer_res = self.maxpool(layer)
        cls_layer_res = self.head_layer(cls_layer_res)
        cls_layer_res = cls_layer_res.view(cls_layer_res.size(0), -1)
        # cls_layer_res = self.head_layer(cls_layer_res)

        cams_dict = {
            "main-relu-cam": F.relu(orig_cams[:, 1:]),
            "clip-l3-relu-sim": F.relu(clip_cams_3[:, 1:]),
            "clip-l4-relu-sim": F.relu(clip_cams_4[:, 1:]),
            "layer-relu-cam": F.relu(layer_cams[:, 1:]),
        }

        cls_pred_dicts = {
            "clip-l3-cls": clip_cls_res_3,
            "clip-l4-cls": clip_cls_res_4,
            "layer-cls": cls_layer_res,
            # "l-full-cls": cls_layer_res,
            "main-cls": cls_res,
        }

        constraint_dict = {
            # "main-cam": orig_cams,
            # # "clip-l3-sim": clip_cams_3,
            # "clip-l4-sim": clip_cams_4,
            # "layer-cam": layer_cams,
        }

        return cls_pred_dicts, cams_dict, constraint_dict

    def forward_dual_branches(self, input_data):
        org = input_data[:, :3, :, :]
        layer = input_data[:, 3:6, :, :]

        org, layer = self.forward_both_stage1(org, layer)
        org, layer = self.forward_both_stage2(org, layer)
        org, layer = self.forward_both_stage3(org, layer)
        org, layer = self.forward_both_stage4(org, layer)

        orig_cams = F.conv2d(org, self.head_org.weight)
        layer_cams = F.conv2d(
            layer, self.head_layer.weight
        )  # .unsqueeze(-1).unsqueeze(-1))
        if not self.constraint_loss:
            orig_cams = orig_cams.detach()
            layer_cams = layer_cams.detach()

        cls_res = self.maxpool(org)
        cls_res = self.head_org(cls_res)
        cls_res = cls_res.view(cls_res.size(0), -1)

        cls_layer_res = self.maxpool(layer)
        cls_layer_res = self.head_layer(cls_layer_res)
        cls_layer_res = cls_layer_res.view(cls_layer_res.size(0), -1)

        cams_dict = {
            "main-relu-cam": F.relu(orig_cams[:, 1:]),
            "layer-relu-cam": F.relu(layer_cams[:, 1:]),
        }

        cls_pred_dicts = {
            "layer-cls": cls_layer_res,
            # "l-full-cls": cls_layer_res,
            "main-cls": cls_res,
        }

        constraint_dict = {
            # "main-cam": orig_cams,
            # # "clip-l3-sim": clip_cams_3,
            # "clip-l4-sim": clip_cams_4,
            # "layer-cam": layer_cams,
        }
        return cls_pred_dicts, cams_dict, constraint_dict

    def forward_clip_only_branches(self, input_data):
        org = input_data[:, :3, :, :]
        # clip features
        clip_txt_f = self.clip_f.to(org.device)  # cls, 768

        org, attns = self.encoder_org(org)

        l_fea_4 = self.clip_fc4(clip_txt_f)  # num_cls, 512
        clip_cams_4, clip_cls_res_4 = self.get_sim_stage4(
            org[-1], l_fea_4, self.logit_scale4
        )

        l_fea_3 = self.clip_fc3(clip_txt_f)  # num_cls, 512
        clip_cams_3, clip_cls_res_3 = self.get_sim_stage3(
            org[-2], l_fea_3, self.logit_scale3
        )

        orig_cams = F.conv2d(org[-1], self.head_org.weight)
        if not self.constraint_loss:
            orig_cams = orig_cams.detach()
            clip_cams_3 = clip_cams_3.detach()
            clip_cams_4 = clip_cams_4.detach()

        cls_res = self.maxpool(org[-1])
        cls_res = self.head_org(cls_res)
        cls_res = cls_res.view(cls_res.size(0), -1)

        cams_dict = {
            "main-relu-cam": F.relu(orig_cams[:, 1:]),
            "clip-l3-relu-sim": F.relu(clip_cams_3[:, 1:]),
            "clip-l4-relu-sim": F.relu(clip_cams_4[:, 1:]),
        }

        cls_pred_dicts = {
            "clip-l3-cls": clip_cls_res_3,
            "clip-l4-cls": clip_cls_res_4,
            "main-cls": cls_res,
        }
        return cls_pred_dicts, cams_dict, {}

    def forward_org_branch(self, input_data, caption_input=None):
        org = input_data[:, :3, :, :]
        org, attns = self.encoder_org(org)
        last_org = org[-1]
        if self.caption_branch:
            # caption_input: batch, 512
            caption_input = caption_input.to(last_org.device)
            last_org = self.fuse_caption_feature(last_org, caption_input)

        orig_cams = F.conv2d(last_org, self.head_org.weight)
        if not self.constraint_loss:
            orig_cams = orig_cams.detach()

        cls_res = self.maxpool(last_org)
        cls_res = self.head_org(cls_res)
        cls_res = cls_res.view(cls_res.size(0), -1)
        cams_dict = {
            "main-relu-cam": F.relu(orig_cams[:, 1:]),
        }
        cls_pred_dicts = {
            "main-cls": cls_res,
        }

        constraint_dict = {}

        return cls_pred_dicts, cams_dict, constraint_dict

    def forward(self, input_data, caption_input=None):
        if self.layer_branch and self.clip_f is not None:
            return self.forward_dual_clip_branches(input_data, caption_input)
        elif self.layer_branch:
            return self.forward_dual_branches(input_data)
        elif self.clip_f is not None:
            return self.forward_clip_only_branches(input_data)
        else:
            return self.forward_org_branch(input_data, caption_input)
