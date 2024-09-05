from torch import nn
import torch
from mmdet.models.backbones.resnet import BasicBlock

import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule

class ConvBasic(BaseModule):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None):
        super(ConvBasic, self).__init__(init_cfg)
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 bias=True,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class LabelBackbone(nn.Module):
    def __init__(self,
                 label_features,
                 box_features,
                 hidden_features,
                 out_features,
                 stride,
                 feature_size,
                 ):
        super().__init__()

        self.mlp_box = Mlp(box_features, hidden_features, hidden_features)
        self.mlp_label = Mlp(label_features, hidden_features, hidden_features)
        self.mlp2 = Mlp(hidden_features, hidden_features, hidden_features)
        self.ln2 = nn.LayerNorm(hidden_features)
        self.conv1 = BasicBlock(hidden_features, hidden_features)
        # self.output_norm = nn.LayerNorm([hidden_features, feature_size, feature_size])
        self.relu = nn.ReLU(inplace=True)

        conv_blocks = []
        for i in range(len(out_features)):
            conv_block=[
                nn.Conv2d(hidden_features,
                          out_features[i],
                          kernel_size=3,
                          stride=stride[i],
                          padding=1,
                          padding_mode='replicate')
                ]

            conv_block.append(nn.LayerNorm([out_features[i], feature_size//stride[i], feature_size//stride[i]]))
            conv_block.append(nn.ReLU(inplace=True))
            conv_block = nn.Sequential(*conv_block)
            conv_blocks.append(conv_block)

        self.conv_blocks = nn.ModuleList(conv_blocks)


    def forward(self, box, label, bev_mask):
        B,W,H,C = box.shape

        box = box.permute(0,3,1,2).contiguous()
        label = label.permute(0,3,1,2).contiguous()

        box = box.flatten(2).transpose(1, 2)
        label = label.flatten(2).transpose(1, 2)

        box = self.mlp_box(box)
        label = self.mlp_label(label)
        out = self.mlp2(box+label)
        out = self.ln2(out)

        out = out.permute(0,2,1).contiguous()
        out = out.view(B, -1, W, H).contiguous()

        out = self.conv1(out)
        outs = []
        for i in range(len(self.conv_blocks)):
            outs.append(self.conv_blocks[i](out))

        return outs

