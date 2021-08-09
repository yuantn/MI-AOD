import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
from mmdet.core import (build_anchor_generator, build_assigner,
                        build_bbox_coder, build_sampler, multi_apply)
from ..builder import HEADS
from ..losses import smooth_l1_loss
from .MIAOD_head import MIAODHead
import numpy as np


# TODO: add loss evaluator for SSD
@HEADS.register_module()
class SSDHead(MIAODHead):
    """SSD head used in https://arxiv.org/abs/1512.02325.

    Args:
        C (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        anchor_generator (dict): Config dict for anchor generator
        background_label (int | None): Label ID of background, set as 0 for
            RPN and C for other heads. It will automatically set as
            C if None is given.
        bbox_coder (dict): Config of bounding box coder.
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied on decoded bounding boxes. Default: False
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
    """  # noqa: W605

    def __init__(self, C=20, in_channels=(512, 1024, 512, 256, 256, 256),
                 anchor_generator=dict(type='SSDAnchorGenerator', scale_major=False, input_size=300,
                                       strides=[8, 16, 32, 64, 100, 300],
                                       ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2]),
                                       basesize_ratio_range=(0.1, 0.9)),
                 background_label=20,
                 bbox_coder=dict(type='DeltaXYWHBBoxCoder', target_means=[.0, .0, .0, .0],
                                 target_stds=[1.0, 1.0, 1.0, 1.0]),
                 reg_decoded_bbox=False, train_cfg=None, test_cfg=None):
        super(MIAODHead, self).__init__()
        if train_cfg is not None:
            self.param_lambda = train_cfg.param_lambda
        self.in_channels = in_channels
        self.C = C
        self.cls_out_channels = C + 1  # add background class
        self.anchor_generator = build_anchor_generator(anchor_generator)
        N = self.anchor_generator.num_base_anchors
        self.l_imgcls = nn.BCELoss()

        f_r_convs = []
        f_1_convs = []
        f_2_convs = []
        f_mil_convs = []
        for i in range(len(in_channels)):
            f_r_convs.append(nn.Conv2d(in_channels[i], N[i] * 4, kernel_size=3, padding=1))
            f_1_convs.append(nn.Conv2d(in_channels[i], N[i] * (C + 1), kernel_size=3, padding=1))
            f_2_convs.append(nn.Conv2d(in_channels[i], N[i] * (C + 1), kernel_size=3, padding=1))
            f_mil_convs.append(nn.Conv2d(in_channels[i], N[i] * (C + 1), kernel_size=3, padding=1))
        self.f_r_convs = nn.ModuleList(f_r_convs)
        self.f_1_convs = nn.ModuleList(f_1_convs)
        self.f_2_convs = nn.ModuleList(f_2_convs)
        self.f_mil_convs = nn.ModuleList(f_mil_convs)
        self.background_label = (C if background_label is None else background_label)
        # background_label should be either 0 or C
        assert (self.background_label == 0 or self.background_label == C)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.reg_decoded_bbox = reg_decoded_bbox
        self.use_sigmoid_cls = False
        self.cls_focal_loss = False
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # set sampling=False for archor_target
        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # SSD sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.fp16_enabled = False

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)

    def forward(self, x):
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                y_f (list[Tensor]): Classification scores for all scale
                    levels, each is a 4D-tensor, the channels number is
                    N * C.
                y_head_f_r (list[Tensor]): Box energies / deltas for all scale
                    levels, each is a 4D-tensor, the channels number is
                    N * 4.
        """

        #  add another cls
        y_head_f_1 = []
        y_head_f_2 = []
        y_head_f_r = []
        y_head_cls = []
        for x_single, reg_conv, cls_conv1, cls_conv2, mil_conv in \
                zip(x, self.f_r_convs, self.f_1_convs, self.f_2_convs, self.f_mil_convs):
            y_head_f_1_single = cls_conv1(x_single)
            y_head_f_2_single = cls_conv2(x_single)
            y_head_f_r_single = reg_conv(x_single)
            y_head_f_mil = mil_conv(x_single)

            y_head_cls_term2 = (y_head_f_1_single + y_head_f_2_single) / 2
            y_head_cls_term2 = y_head_cls_term2.detach()
            y_head_f_mil = y_head_f_mil.permute(0, 2, 3, 1).reshape(y_head_f_1_single.shape[0],
                                                                    -1, self.cls_out_channels)
            y_head_cls_term2 = y_head_cls_term2.permute(0, 2, 3, 1).reshape(y_head_f_1_single.shape[0],
                                                                            -1, self.cls_out_channels)
            y_head_cls_single = y_head_f_mil.softmax(2) * y_head_cls_term2.sigmoid().max(2, keepdim=True)[0].softmax(1)

            y_head_f_1.append(y_head_f_1_single)
            y_head_f_2.append(y_head_f_2_single)
            y_head_f_r.append(y_head_f_r_single)
            y_head_cls.append(y_head_cls_single)
        return y_head_f_1, y_head_f_2, y_head_f_r, y_head_cls

    def l_det(self, y_head_f_single, y_head_f_r_single, x_i_single, y_cls_single, label_weights,
                    y_loc_single, bbox_weights, num_total_samples):
        """Compute loss of a single image.

        Args:
            y_head_f_single (Tensor): Box scores for eachimage
                Has shape (num_total_anchors, C).
            y_head_f_r_single (Tensor): Box energies / deltas for each image
                level with shape (num_total_anchors, 4).
            x_i_single (Tensor): Box reference for each scale level with shape
                (num_total_anchors, 4).
            y_cls_single (Tensor): Labels of each anchors with shape
                (num_total_anchors,).
            label_weights (Tensor): Label weights of each anchor with shape
                (num_total_anchors,)
            y_loc_single (Tensor): BBox regression targets of each anchor wight
                shape (num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (num_total_anchors, 4).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        l_det_cls_all = F.cross_entropy(
            y_head_f_single, y_cls_single, reduction='none') * label_weights

        # FG cat_id: [0, C -1], BG cat_id: C
        pos_inds = ((y_cls_single >= 0) & (y_cls_single < self.background_label)).nonzero().reshape(-1)
        neg_inds = (y_cls_single == self.background_label).nonzero().view(-1)

        # fore/background partition
        if pos_inds.dim() == 0:
            return l_det_cls_all.sum()[None]*0, l_det_cls_all.sum()
        num_pos_samples = pos_inds.size(0)
        num_neg_samples = self.train_cfg.neg_pos_ratio * num_pos_samples
        if num_neg_samples > neg_inds.size(0):
            num_neg_samples = neg_inds.size(0)
        topk_l_det_cls_neg, _ = l_det_cls_all[neg_inds].topk(num_neg_samples)
        l_det_cls_pos = l_det_cls_all[pos_inds].sum()
        l_det_cls_neg = topk_l_det_cls_neg.sum()
        # loss for pos and neg (our loss only use pos)
        l_det_cls = (l_det_cls_pos + l_det_cls_neg) / num_total_samples

        if self.reg_decoded_bbox:
            y_head_f_r_single = self.bbox_coder.decode(x_i_single, y_head_f_r_single)
        l_det_loc = smooth_l1_loss(y_head_f_r_single, y_loc_single, bbox_weights,
                                   beta=self.train_cfg.smoothl1_beta, avg_factor=num_total_samples)
        return l_det_cls[None], l_det_loc

    def L_det(self, y_f, y_f_r, y_head_cls, y_loc_img, y_cls_img, img_metas, y_loc_img_ignore=None):
        """Compute losses of the head.

        Args:
            y_f (list[Tensor]): Box scores for each scale level
                Has shape (N, N * C, H, W)
            y_f_r (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, N * 4, H, W)
            y_loc_img (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            y_cls_img (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            y_loc_img_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in y_f]
        assert len(featmap_sizes) == self.anchor_generator.num_levels
        device = y_f[0].device
        x_i, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, device=device)
        cls_reg_targets = self.get_targets(x_i, valid_flag_list, y_loc_img, img_metas,
                                           y_loc_img_ignore_list=y_loc_img_ignore,
                                           y_cls_img_list=y_cls_img,
                                           label_channels=1, unmap_outputs=False)
        if cls_reg_targets is None:
            return None
        (y_cls, label_weights_list, y_loc, bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets
        num_images = len(img_metas)
        all_y_f = torch.cat([s.permute(0, 2, 3, 1).reshape(num_images, -1, self.cls_out_channels) for s in y_f], 1)
        all_y_cls = torch.cat(y_cls, -1).view(num_images, -1)
        all_label_weights = torch.cat(label_weights_list, -1).view(num_images, -1)
        all_y_f_r = torch.cat([b.permute(0, 2, 3, 1).reshape(num_images, -1, 4) for b in y_f_r], -2)
        all_y_loc = torch.cat(y_loc, -2).view(num_images, -1, 4)
        all_bbox_weights = torch.cat(bbox_weights_list, -2).view(num_images, -1, 4)
        # concat all level anchors to a single tensor
        all_x_i = []
        for i in range(num_images):
            all_x_i.append(torch.cat(x_i[i]))
        # check NaN and Inf
        assert torch.isfinite(all_y_f).all().item(), 'classification scores become infinite or NaN!'
        assert torch.isfinite(all_y_f_r).all().item(), 'bbox predications become infinite or NaN!'
        l_det_cls, l_det_loc = multi_apply(self.l_det, all_y_f, all_y_f_r, all_x_i,
                                             all_y_cls, all_label_weights, all_y_loc, all_bbox_weights,
                                             num_total_samples=num_total_pos)
        # compute mil loss
        y_head_cls_1level, y_cls_1level = self.get_img_gtlabel_score(y_cls_img, y_head_cls)
        l_imgcls = self.l_imgcls(y_head_cls_1level, y_cls_1level)
        return dict(l_det_cls=l_det_cls, l_det_loc=l_det_loc, l_imgcls=[l_imgcls])

    def l_wave_dis(self, y_head_f_1_single, y_head_f_2_single, y_head_cls_single):
        w_i = y_head_cls_single.detach()
        l_det_cls_all = (abs(y_head_f_1_single.softmax(-1) - y_head_f_2_single.softmax(-1)) *
                         w_i.reshape(-1, self.cls_out_channels)).mean(dim=1).sum() * self.param_lambda
        l_det_loc = torch.tensor([0.0], device=y_head_f_1_single.device)
        return l_det_cls_all[None], l_det_loc

    # Re-weighting and minimizing instance uncertainty
    def L_wave_min(self, y_f, y_f_r, y_head_cls, y_loc_img, y_cls_img, img_metas, y_loc_img_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in y_f[0]]
        assert len(featmap_sizes) == self.anchor_generator.num_levels
        device = y_f[0][0].device
        x_i, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, device=device)
        cls_reg_targets = self.get_targets(x_i, valid_flag_list, y_loc_img, img_metas,
                                           y_loc_img_ignore_list=y_loc_img_ignore,
                                           y_cls_img_list=y_cls_img,
                                           label_channels=1, unmap_outputs=False)
        if cls_reg_targets is None:
            return None
        (y_cls, label_weights_list, y_loc, bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets
        num_images = len(img_metas)
        all_y_f_1 = torch.cat([s.permute(0, 2, 3, 1).reshape(num_images, -1, self.cls_out_channels) for s in y_f[0]], 1)
        all_y_f_2 = torch.cat([s.permute(0, 2, 3, 1).reshape(num_images, -1, self.cls_out_channels) for s in y_f[1]], 1)
        all_y_cls = torch.cat(y_cls, -1).view(num_images, -1)
        all_label_weights = torch.cat(label_weights_list, -1).view(num_images, -1)
        all_y_f_r = torch.cat([b.permute(0, 2, 3, 1).reshape(num_images, -1, 4) for b in y_f_r], -2)
        all_y_loc = torch.cat(y_loc, -2).view(num_images, -1, 4)
        all_bbox_weights = torch.cat(bbox_weights_list, -2).view(num_images, -1, 4)
        # concat all level anchors to a single tensor
        all_x_i = []
        for i in range(num_images):
            all_x_i.append(torch.cat(x_i[i]))
        all_y_head_cls = torch.cat([s for s in y_head_cls], 1)
        l_wave_dis, l_det_loc = multi_apply(self.l_wave_dis, all_y_f_1, all_y_f_2, all_y_head_cls)
        if np.array([y_loc_img[i].sum() for i in range(len(y_loc_img))]).sum() < 0:
            l_det_cls = [torch.tensor(0.0, device=device)]
            l_det_loc = [torch.tensor(0.0, device=device)]
            for (i, value) in enumerate(l_det_loc):
                if value.isnan():
                    l_det_loc[i].data = torch.tensor(0.0, device=device)
            # compute mil loss
            y_head_cls_1level, y_pseudo = self.get_img_pseudolabel_score(y_f, y_head_cls)
            if (y_pseudo.sum(1) == 0).sum() > 0:  # ignore hard images
                l_imgcls = self.l_imgcls(y_head_cls_1level, y_pseudo) * 0
            else:
                l_imgcls = self.l_imgcls(y_head_cls_1level, y_pseudo)
        else:
            l_det_cls1, l_det_loc1 = multi_apply(self.l_det, all_y_f_1, all_y_f_r, all_x_i,
                                                 all_y_cls, all_label_weights, all_y_loc, all_bbox_weights,
                                                 num_total_samples=num_total_pos)
            l_det_cls2, l_det_loc2 = multi_apply(self.l_det, all_y_f_2, all_y_f_r, all_x_i,
                                                 all_y_cls, all_label_weights, all_y_loc, all_bbox_weights,
                                                 num_total_samples=num_total_pos)
            l_det_cls = list(map(lambda m, n: (m + n) / 2, l_det_cls1, l_det_cls2))
            l_det_loc = list(map(lambda m, n: (m + n) / 2, l_det_loc1, l_det_loc2))
            l_wave_dis = list(map(lambda m: m * 0.0, l_wave_dis))
            # compute mil loss
            y_head_cls_1level, y_cls_1level = self.get_img_gtlabel_score(y_cls_img, y_head_cls)
            l_imgcls = self.l_imgcls(y_head_cls_1level, y_cls_1level)
        return dict(l_det_cls=l_det_cls, l_det_loc=l_det_loc, l_wave_dis=l_wave_dis, l_imgcls=[l_imgcls])

    def l_wave_dis_minus(self, y_head_f_1_single, y_head_f_2_single, y_head_cls_single):
        w_i = y_head_cls_single.detach()
        l_det_cls_all = ((1 - abs(y_head_f_1_single.softmax(-1) - y_head_f_2_single.softmax(-1))) *
                         w_i.reshape(-1, self.cls_out_channels)).mean(dim=1).sum() * self.param_lambda
        l_det_loc = torch.tensor([0.0], device=y_head_f_1_single.device)
        return l_det_cls_all[None], l_det_loc

    # Re-weighting and maximizing instance uncertainty
    def L_wave_max(self, y_f, y_f_r, y_head_cls, y_loc_img, y_cls_img, img_metas, y_loc_img_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in y_f[0]]
        assert len(featmap_sizes) == self.anchor_generator.num_levels
        device = y_f[0][0].device
        x_i, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, device=device)
        cls_reg_targets = self.get_targets(x_i, valid_flag_list, y_loc_img, img_metas,
                                           y_loc_img_ignore_list=y_loc_img_ignore,
                                           y_cls_img_list=y_cls_img,
                                           label_channels=1, unmap_outputs=False)
        if cls_reg_targets is None:
            return None
        (y_cls, label_weights_list, y_loc, bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets
        num_images = len(img_metas)
        all_y_f_1 = torch.cat([s.permute(0, 2, 3, 1).reshape(num_images, -1, self.cls_out_channels) for s in y_f[0]], 1)
        all_y_f_2 = torch.cat([s.permute(0, 2, 3, 1).reshape(num_images, -1, self.cls_out_channels) for s in y_f[1]], 1)
        all_y_cls = torch.cat(y_cls, -1).view(num_images, -1)
        all_label_weights = torch.cat(label_weights_list, -1).view(num_images, -1)
        all_y_f_r = torch.cat([b.permute(0, 2, 3, 1).reshape(num_images, -1, 4) for b in y_f_r], -2)
        all_y_loc = torch.cat(y_loc, -2).view(num_images, -1, 4)
        all_bbox_weights = torch.cat(bbox_weights_list, -2).view(num_images, -1, 4)
        # concat all level anchors to a single tensor
        all_x_i = []
        for i in range(num_images):
            all_x_i.append(torch.cat(x_i[i]))
        all_y_head_cls = torch.cat([s for s in y_head_cls], 1)
        l_wave_dis_minus, l_det_loc = multi_apply(self.l_wave_dis_minus, all_y_f_1, all_y_f_2, all_y_head_cls)
        if np.array([y_loc_img[i].sum() for i in range(len(y_loc_img))]).sum() < 0:
            l_det_cls = [torch.tensor(0.0, device=device)]
            l_det_loc = [torch.tensor(0.0, device=device)]
            for (i, value) in enumerate(l_det_loc):
                if value.isnan():
                    l_det_loc[i].data = torch.tensor(0.0, device=device)
            l_imgcls = torch.tensor(0.0, device=device)
        else:
            l_det_cls1, l_det_loc1 = multi_apply(self.l_det, all_y_f_1, all_y_f_r, all_x_i,
                                                 all_y_cls, all_label_weights, all_y_loc, all_bbox_weights,
                                                 num_total_samples=num_total_pos)
            l_det_cls2, l_det_loc2 = multi_apply(self.l_det, all_y_f_2, all_y_f_r, all_x_i,
                                                 all_y_cls, all_label_weights, all_y_loc, all_bbox_weights,
                                                 num_total_samples=num_total_pos)
            l_det_cls = list(map(lambda m, n: (m + n) / 2, l_det_cls1, l_det_cls2))
            l_det_loc = list(map(lambda m, n: (m + n) / 2, l_det_loc1, l_det_loc2))
            l_wave_dis_minus = list(map(lambda m: m * 0.0, l_wave_dis_minus))
            # compute mil loss
            y_head_cls_1level, y_cls_1level = self.get_img_gtlabel_score(y_cls_img, y_head_cls)
            l_imgcls = self.l_imgcls(y_head_cls_1level, y_cls_1level)
        return dict(l_det_cls=l_det_cls, l_det_loc=l_det_loc, l_wave_dis_minus=l_wave_dis_minus, l_imgcls=[l_imgcls])
