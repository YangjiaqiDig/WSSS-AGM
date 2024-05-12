import torch.nn.functional as F
import torch
import torch.nn as nn


def dsc_loss(y_pred, y_true, varepsilon=1.0e-8):
    epsilon = 1.0e-8
    y_true = y_true.float()
    y_pred = torch.clamp(y_pred, epsilon, 1.0 - epsilon)

    numerator = 2 * (y_true * y_pred * (1 - y_pred)).sum() + varepsilon
    denominator = (y_true + y_pred * (1 - y_pred)).sum() + varepsilon

    return 1 - numerator / denominator


def tversky_loss(y_pred, y_true, alpha=0.7, beta=0.3, varepsilon=1.0e-8):
    epsilon = 1.0e-8
    y_true = y_true.float()
    y_pred = torch.clamp(y_pred, epsilon, 1.0 - epsilon)

    numerator = (y_true * y_pred).sum() + varepsilon
    denominator = (
        y_true.sum()
        + alpha * (1 - y_true).sum()
        + beta * (1 - y_pred).sum()
        + varepsilon
    )

    return 1 - numerator / denominator


def max_norm(p, e=1e-5):
    if p.dim() == 3:
        C, H, W = p.size()
        p = F.relu(p)
        max_v = torch.max(p.view(C, -1), dim=-1)[0].view(C, 1, 1)
        min_v = torch.min(p.view(C, -1), dim=-1)[0].view(C, 1, 1)
        p = F.relu(p - min_v - e) / (max_v - min_v + e)
    elif p.dim() == 4:
        N, C, H, W = p.size()
        p = F.relu(p)
        max_v = torch.max(p.view(N, C, -1), dim=-1)[0].view(N, C, 1, 1)
        min_v = torch.min(p.view(N, C, -1), dim=-1)[0].view(N, C, 1, 1)
        p = F.relu(p - min_v - e) / (max_v - min_v + e)

    return p


class mIoULoss(nn.Module):
    def __init__(self, n_foreground=2):
        super(mIoULoss, self).__init__()
        self.n_foreground = n_foreground

    def forward(self, inputs, target, cls_label):
        # inputs => N x n_foreground x H x W
        # target => N x 3 x H x W
        N = inputs.size()[0]
        # extend the target 1st dimension to match the input
        target_extend = (
            target[:, 0, :, :]
            .unsqueeze(1)
            .expand(N, self.n_foreground, target.shape[-2], target.shape[-1])
        )
        target_bg = 1 - target[:, 0]
        target_extend = torch.cat((target_extend, target_bg.unsqueeze(1)), dim=1)
        assert list(inputs.shape) == list(
            target_extend.shape
        ), "target sizes must be b, 3, h, w."

        norm_inputs = max_norm(inputs)
        cls_mask = (
            cls_label.unsqueeze(2)
            .unsqueeze(3)
            .expand(N, self.n_foreground + 1, target.shape[-2], target.shape[-1])
        )
        target_extend_with_cls = target_extend * cls_mask
        # import pdb; pdb.set_trace()

        # Numerator Product
        numerator_p = norm_inputs * target_extend_with_cls  # target_extend
        ## Sum over all pixels N x C x H x W => N x C
        inter = numerator_p.view(N, self.n_foreground + 1, -1).sum(2)

        # Denominator
        union_bg = (
            norm_inputs[:, -1]
            + target_extend[:, -1]
            - (norm_inputs[:, -1] * target_extend[:, -1])
        )
        union_bg = union_bg.view(N, 1, -1).sum(2)
        loss_bg = inter[:, -1] / (union_bg + 1e-8)
        union_fg = norm_inputs[
            :, :-1
        ]  # + target_extend[:,:-1] - (norm_inputs[:,:-1] * target_extend[:,:-1])
        union_fg = union_fg.view(N, self.n_foreground, -1).sum(2)
        loss_fg = inter[:, :-1] / (union_fg + 1e-8)
        loss = torch.mean(loss_bg) + torch.mean(loss_fg)
        ## Sum over all pixels N x C x H x W => N x C
        # union = union.view(N, self.n_foreground+1, -1).sum(2)
        # union = norm_inputs + target_extend - (norm_inputs * target_extend)
        # loss = (inter) / (union + 1e-8)

        if -torch.mean(loss) > 0:
            import pdb

            pdb.set_trace()

        # foreground and background cannot overlap, loss
        # bg_extend = norm_inputs[:,-1:].expand(N, self.n_foreground, target.shape[-2], target.shape[-1])
        # fb_inter = torch.sum(norm_inputs[:,:-1] * bg_extend, dim=[2,3])
        # fb_union = torch.sum((norm_inputs[:,:-1] + bg_extend), dim=[2,3])
        # fb_loss = torch.mean((2*fb_inter) / (fb_union + 1e-8))
        # fb_loss = 0

        ## Return average loss over n_foreground and batch
        return -torch.mean(loss)  # + fb_loss


def equivariant_regularization(x, y):
    x = x.view(x.size(0), -1)
    y = y.view(y.size(0), -1)
    return torch.mean(torch.sum((x - y) ** 2, dim=1))


def er1_loss(x, y):
    return torch.mean(torch.abs(x - y))
