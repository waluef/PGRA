"""Rotational convolutional layer."""

import torch
from torch import nn

from ..ops.functions import ModulatedDeformConvFunction
from ..ops.modules import ModulatedDeformConv


class RotationConvLayer(ModulatedDeformConv):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, groups=1, deformable_groups=1,
                 im2col_step=64, bias=True):
        super(RotationConvLayer, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, deformable_groups, im2col_step, bias,
        )
        mask_channels = (
            self.deformable_groups * self.kernel_size[0] * self.kernel_size[1])
        self.conv_mask = nn.Conv2d(
            self.in_channels, mask_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self._init_mask()

    def _init_mask(self):
        self.conv_mask.weight.data.zero_()
        self.conv_mask.bias.data.zero_()

    def _gene_offset(self, b, h, w, angle):
        x_v = (self.kernel_size[0] - 1) // 2
        y_v = (self.kernel_size[1] - 1) // 2
        x_axis = torch.arange(-x_v, x_v + 1)
        y_axis = torch.arange(-y_v, y_v + 1)
        x_coor, y_coor = torch.meshgrid(x_axis, y_axis)
        x_coor = x_coor.float().contiguous().view(-1, 1)
        y_coor = y_coor.float().contiguous().view(-1, 1)
        coor = torch.cat((x_coor, y_coor), dim=1).unsqueeze(2).to(angle.device)

        out_h = (h + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        out_w = (w + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        start_h = self.kernel_size[0] // 2 - self.padding[0]
        start_w = self.kernel_size[1] // 2 - self.padding[1]
        angle = angle[:, :, start_h:start_h + out_h, start_w:start_w + out_w]

        cos_theta = torch.cos(angle).unsqueeze(-1)
        sin_theta = torch.sin(angle).unsqueeze(-1)
        rot_minus_i = torch.cat(
            (cos_theta - 1, sin_theta, -sin_theta, cos_theta - 1), dim=-1)
        rot_minus_i = rot_minus_i.contiguous().view(-1, 1, 2, 2)

        offset = torch.matmul(rot_minus_i, coor)
        offset = offset.reshape(b, out_h, out_w, -1).permute(0, 3, 1, 2).contiguous()
        return offset

    def forward(self, input, angle=None, offset=None, mask=None):
        b, _, h, w = input.size()

        if angle is None:
            angle = torch.zeros_like(input)[:, :1, :, :]
        elif isinstance(angle, (int, float)):
            angle = torch.deg2rad(torch.tensor(float(angle)))
            angle = angle.view(1, 1, 1, 1).repeat(b, 1, h, w).to(input.device)

        if offset is None:
            offset = self._gene_offset(b, h, w, angle).detach()

        if mask is None:
            mask = torch.sigmoid(self.conv_mask(input))

        return ModulatedDeformConvFunction.apply(
            input, offset, mask, self.weight, self.bias,
            self.stride, self.padding, self.dilation,
            self.groups, self.deformable_groups, self.im2col_step,
        )
