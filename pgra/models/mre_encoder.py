"""MRE encoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rcl import RotationConvLayer


class _MultiAngleRCLBranch(nn.Module):
    def __init__(self, in_channels, out_channels, angle_range, kernel_size=3):
        super(_MultiAngleRCLBranch, self).__init__()
        self.angle_range = list(angle_range)
        self.rot_conv = RotationConvLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.fuse = nn.Conv2d(
            in_channels * len(self.angle_range), out_channels, kernel_size=1)

    def forward(self, x):
        feats = [self.rot_conv(x, angle=a) for a in self.angle_range]
        concat = torch.cat(feats, dim=1)
        return self.fuse(concat)


class DFB(_MultiAngleRCLBranch):
    pass


class PPB(_MultiAngleRCLBranch):
    pass


class SFFM(nn.Module):
    def __init__(self, channels):
        super(SFFM, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, f_asc, f_data):
        return self.block(f_asc + f_data)


class MREEncoder(nn.Module):
    def __init__(self, in_channels, part_num, angle_range):
        super(MREEncoder, self).__init__()
        self.dfb = DFB(in_channels, in_channels, angle_range)
        self.ppb = PPB(part_num, in_channels, angle_range)
        self.sffm = SFFM(in_channels)

    def forward(self, x_data, x_asc):
        if x_asc.shape[-2:] != x_data.shape[-2:]:
            x_asc = F.interpolate(
                x_asc, size=x_data.shape[-2:],
                mode="bilinear", align_corners=False,
            )
        f_data = self.dfb(x_data)
        f_asc = self.ppb(x_asc)
        return self.sffm(f_asc, f_data)
