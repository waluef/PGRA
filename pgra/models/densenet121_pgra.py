"""DenseNet-121 backbone."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from .pgra import PGRA, Identity


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated = torch.cat(inputs, 1)
        return conv(relu(norm(concated)))
    return bn_function


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size,
                 drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module("norm1", nn.BatchNorm2d(num_input_features))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv1", nn.Conv2d(
            num_input_features, bn_size * growth_rate,
            kernel_size=1, stride=1, bias=False))
        self.add_module("norm2", nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module("conv2", nn.Conv2d(
            bn_size * growth_rate, growth_rate,
            kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def forward(self, *prev_features):
        bn_fn = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.memory_efficient and any(p.requires_grad for p in prev_features):
            bottleneck = cp.checkpoint(bn_fn, *prev_features)
        else:
            bottleneck = bn_fn(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck)))
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training)
        return new_features


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
                 drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for _, layer in self.named_children():
            features.append(layer(*features))
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module("norm", nn.BatchNorm2d(num_input_features))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(
            num_input_features, num_output_features,
            kernel_size=1, stride=1, bias=False))
        self.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet121PGRA(nn.Module):
    def __init__(self, num_classes=10,
                 attention_setting=True,
                 angle_range=None,
                 part_num=4,
                 growth_rate=32,
                 block_config=(6, 12, 24, 16),
                 num_init_features=64,
                 bn_size=4,
                 drop_rate=0.0,
                 memory_efficient=False):
        super(DenseNet121PGRA, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv2d(1, num_init_features, kernel_size=7,
                                stride=2, padding=3, bias=False)),
            ("norm0", nn.BatchNorm2d(num_init_features)),
            ("relu0", nn.ReLU(inplace=True)),
            ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        num_features = num_init_features
        self.attn = nn.ModuleList()
        self.block = nn.ModuleList()
        self.trans = nn.ModuleList()

        def _make_attn(channels):
            return (PGRA(channels, part_num=part_num, angle_range=angle_range)
                    if attention_setting else Identity())

        for i, num_layers in enumerate(block_config):
            self.attn.append(_make_attn(num_features))
            self.block.append(_DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            ))
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                self.trans.append(_Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2,
                ))
                num_features //= 2

        self.norm5 = nn.BatchNorm2d(num_features)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x, asc_part):
        feats = self.features(x)
        feats = self.attn[0](feats, asc_part)

        for i in range(len(self.trans)):
            feats = self.block[i](feats)
            feats = self.trans[i](feats)
            feats = self.attn[i + 1](feats, asc_part)

        feats = self.block[-1](feats)
        feats = F.relu(self.norm5(feats), inplace=True)
        out = F.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)
        return self.classifier(out)


def densenet121_pgra(num_classes, **kwargs):
    return DenseNet121PGRA(
        num_classes=num_classes,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
        **kwargs,
    )
