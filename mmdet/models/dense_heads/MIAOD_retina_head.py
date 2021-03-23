import torch.nn as nn
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init
from ..builder import HEADS
from .MIAOD_head import MIAODHead
import torch
from mmdet.core import (build_anchor_generator, build_assigner,
                        build_bbox_coder, build_sampler, multi_apply)


@HEADS.register_module()
class MIAODRetinaHead(MIAODHead):
    r"""An anchor-based head used in `RetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Example:
        >>> import torch
        >>> self = MIAODRetinaHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> y_head_f_i, y_head_f_r = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = y_head_f_i.shape[1] / self.N
        >>> box_per_anchor = y_head_f_r.shape[1] / self.N
        >>> assert cls_per_anchor == (self.C)
        >>> assert box_per_anchor == 4
    """

    def __init__(self, C, in_channels, stacked_convs=4, conv_cfg=None, norm_cfg=None,
                 anchor_generator=dict(type='AnchorGenerator', octave_base_scale=4, cales_per_octave=3,
                                       ratios=[0.5, 1.0, 2.0], strides=[8, 16, 32, 64, 128]), **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super(MIAODRetinaHead, self).__init__(C, in_channels, anchor_generator=anchor_generator, **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.f_1_convs = nn.ModuleList()
        self.f_2_convs = nn.ModuleList()
        self.f_r_convs = nn.ModuleList()
        self.f_mil_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.f_1_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1,
                                             conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
            self.f_2_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1,
                                             conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
            self.f_r_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1,
                                             conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
            self.f_mil_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1,
                                               conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
        self.f_1_retina = nn.Conv2d(self.feat_channels, self.N * self.cls_out_channels, 3, padding=1)
        self.f_2_retina = nn.Conv2d(self.feat_channels, self.N * self.cls_out_channels, 3, padding=1)
        self.f_r_retina = nn.Conv2d(self.feat_channels, self.N * 4, 3, padding=1)
        self.f_mil_retina = nn.Conv2d(self.feat_channels, self.N * self.cls_out_channels, 3, padding=1)

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.f_1_convs:
            normal_init(m.conv, std=0.01)
        for m in self.f_2_convs:
            normal_init(m.conv, std=0.01)
        for m in self.f_r_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.f_1_retina, std=0.01, bias=bias_cls)
        normal_init(self.f_2_retina, std=0.01, bias=bias_cls)
        normal_init(self.f_mil_retina, std=0.01, bias=bias_cls)
        normal_init(self.f_r_retina, std=0.01)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                y_head_f_i (Tensor): Cls scores for a single scale level
                    the channels number is N * C.
                y_head_f_r (Tensor): Box energies / deltas for a single scale
                    level, the channels number is N * 4.
        """
        f_1_feat = x
        f_2_feat = x
        f_r_feat = x
        f_mil_feat = x
        for cls_conv1 in self.f_1_convs:
            f_1_feat = cls_conv1(f_1_feat)
        for cls_conv2 in self.f_2_convs:
            f_2_feat = cls_conv2(f_2_feat)
        for reg_conv in self.f_r_convs:
            f_r_feat = reg_conv(f_r_feat)
        for mil_conv in self.f_mil_convs:
            f_mil_feat = mil_conv(f_mil_feat)
        y_head_f_1 = self.f_1_retina(f_1_feat)
        y_head_f_2 = self.f_2_retina(f_2_feat)
        y_head_f_r = self.f_r_retina(f_r_feat)
        y_head_f_mil = self.f_mil_retina(f_mil_feat)
        y_head_cls_term2 = (y_head_f_1 + y_head_f_2) / 2
        y_head_cls_term2 = y_head_cls_term2.detach()
        y_head_f_mil = y_head_f_mil.permute(0, 2, 3, 1).reshape(y_head_f_1.shape[0], -1, self.cls_out_channels)
        y_head_cls_term2 = y_head_cls_term2.permute(0, 2, 3, 1).reshape(y_head_f_1.shape[0],
                                                                            -1, self.cls_out_channels)
        y_head_cls = y_head_f_mil.softmax(2) * y_head_cls_term2.sigmoid().max(2, keepdim=True)[0].softmax(1)
        return y_head_f_1, y_head_f_2, y_head_f_r, y_head_cls
