"""PGRA block."""

import math

import torch
import torch.nn as nn

from .mre_encoder import MREEncoder


class CIDE(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CIDE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        z = self.avg_pool(x).view(b, c)
        s = self.fc(z).view(b, c, 1, 1)
        return x * s.expand_as(x)


class GPSM(nn.Module):
    def __init__(self, channels):
        super(GPSM, self).__init__()
        self.W_Q = nn.Conv2d(channels, channels, kernel_size=1)
        self.W_K = nn.Conv2d(channels, channels, kernel_size=1)
        self.W_V = nn.Conv2d(channels, channels, kernel_size=1)

        self.W_g = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.W_out = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
        )
        self.register_buffer("scale", torch.tensor(math.sqrt(float(channels))))

    def forward(self, f_phy, f_data):
        b, c, h, w = f_phy.shape

        Q = self.W_Q(f_data).view(b, c, -1).permute(0, 2, 1)
        K = self.W_K(f_phy).view(b, c, -1)
        V = self.W_V(f_phy).view(b, c, -1).permute(0, 2, 1)

        attn = torch.softmax(torch.bmm(Q, K) / self.scale, dim=-1)
        context = torch.bmm(attn, V).permute(0, 2, 1).contiguous()
        context = context.view(b, c, h, w)
        enhanced_phy = self.W_out(context + f_phy)

        alpha = self.W_g(torch.cat([enhanced_phy, f_data], dim=1))
        fused = alpha * enhanced_phy + (1.0 - alpha) * f_data
        return fused, alpha


class PGRA(nn.Module):
    DEFAULT_ANGLE_RANGE = [i * 2.5 for i in range(-5, 6)]  

    def __init__(self, in_channels, part_num,
                 angle_range=None, cide_reduction=2):
        super(PGRA, self).__init__()
        if angle_range is None:
            angle_range = self.DEFAULT_ANGLE_RANGE

        self.pre_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.cide = CIDE(in_channels, reduction=cide_reduction)

        self.mre_encoder = MREEncoder(
            in_channels=in_channels,
            part_num=part_num,
            angle_range=angle_range,
        )

        self.gpsm = GPSM(in_channels)
        self.post_attn = CIDE(in_channels, reduction=16)

    def forward(self, x, asc_part):
        f_data = self.cide(self.pre_conv(x))
        f_phy = self.mre_encoder(x, asc_part)
        fused, _alpha = self.gpsm(f_phy, f_data)
        out = self.post_attn(fused)
        return out + x


class Identity(nn.Module):
    def forward(self, *args):
        return args[0]
