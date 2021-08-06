import torch
import torch.nn as nn
from mmcv.cnn import normal_init
from mmdet.core import (anchor_inside_flags, build_anchor_generator,
                        build_assigner, build_bbox_coder, build_sampler,
                        force_fp32, images_to_levels, multi_apply,
                        multiclass_nms, unmap)
from ..builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead
from mmdet.core import (build_anchor_generator, build_assigner,
                        build_bbox_coder, build_sampler, multi_apply)


class MyEntLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.nn.Softmax(dim=1)(x)
        p = x / torch.repeat_interleave(x.sum(dim=1).unsqueeze(-1), repeats=20, dim=1)
        logp = torch.log2(p)
        ent = -torch.mul(p, logp)
        entloss = torch.sum(ent, dim=1)
        return entloss


@HEADS.register_module()
class MIAODHead(BaseDenseHead):
    """Anchor-based head (RPN, RetinaNet, SSD, etc.).

    Args:
        C (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels. Used in child classes.
        anchor_generator (dict): Config dict for anchor generator
        bbox_coder (dict): Config of bounding box coder.
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied on decoded bounding boxes. Default: False
        background_label (int | None): Label ID of background, set as 0 for
            RPN and C for other heads. It will automatically set as
            C if None is given.
        FL (dict): Config of classification loss.
        SmoothL1 (dict): Config of localization loss.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
    """  # noqa: W605

    def __init__(self, C, in_channels, feat_channels=256,
                 anchor_generator=dict(type='AnchorGenerator', scales=[8, 16, 32],
                                       ratios=[0.5, 1.0, 2.0], strides=[4, 8, 16, 32, 64]),
                 bbox_coder=dict(type='DeltaXYWHBBoxCoder', target_means=(.0, .0, .0, .0),
                                 target_stds=(1.0, 1.0, 1.0, 1.0)),
                 reg_decoded_bbox=False,
                 background_label=None,
                 FL=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                 SmoothL1=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None):
        super(MIAODHead, self).__init__()
        if train_cfg is not None:
            self.param_lambda = train_cfg.param_lambda
        self.in_channels = in_channels
        self.C = C
        self.feat_channels = feat_channels
        self.use_sigmoid_cls = FL.get('use_sigmoid', False)
        # TODO better way to determine whether sample or not
        self.sampling = FL['type'] not in ['FocalLoss', 'GHMC', 'QualityFocalLoss']
        if self.use_sigmoid_cls:
            self.cls_out_channels = C
        else:
            self.cls_out_channels = C + 1
        if self.cls_out_channels <= 0:
            raise ValueError(f'C={C} is too small')
        self.reg_decoded_bbox = reg_decoded_bbox
        self.background_label = (C if background_label is None else background_label)
        # background_label should be either 0 or C
        assert (self.background_label == 0 or self.background_label == C)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.FL = build_loss(FL)
        self.SmoothL1 = build_loss(SmoothL1)
        self.l_imgcls = nn.BCELoss()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # use PseudoSampler when sampling is False
            if self.sampling and hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.fp16_enabled = False
        self.anchor_generator = build_anchor_generator(anchor_generator)
        # usually the numbers of anchors for each level are the same
        # except SSD detectors
        self.N = self.anchor_generator.num_base_anchors[0]
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        self.conv_f_1 = nn.Conv2d(self.in_channels, self.N * self.cls_out_channels, 1)
        self.conv_f_2 = nn.Conv2d(self.in_channels, self.N * self.cls_out_channels, 1)
        self.conv_f_r = nn.Conv2d(self.in_channels, self.N * 4, 1)

    def init_weights(self):
        """Initialize weights of the head."""
        normal_init(self.conv_f_1, std=0.01)
        normal_init(self.conv_f_2, std=0.01)
        normal_init(self.conv_f_r, std=0.01)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                y_head_f_single (Tensor): Cls scores for a single scale level \
                    the channels number is N * C.
                y_head_f_r_single (Tensor): Box energies / deltas for a single scale \
                    level, the channels number is N * 4.
        """
        y_head_f_1_single = self.conv_f_1(x)
        y_head_f_2_single = self.conv_f_2(x)
        y_head_f_r_single = self.conv_f_r(x)
        return y_head_f_1_single, y_head_f_2_single, y_head_f_r_single

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - y_f (list[Tensor]): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is N * C.
                - y_f_r (list[Tensor]): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is N * 4.
        """
        return multi_apply(self.forward_single, feats)

    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            device (torch.device | str): Device for returned tensors

        Returns:
            tuple:
                x_i (list[Tensor]): Anchors of each image.
                valid_flag_list (list[Tensor]): Valid flags of each image.
        """
        num_imgs = len(img_metas)
        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.anchor_generator.grid_anchors(featmap_sizes, device)
        x_i = [multi_level_anchors for _ in range(num_imgs)]
        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.anchor_generator.valid_flags(featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)
        return x_i, valid_flag_list

    def _get_targets_single(self, flat_anchors, valid_flags, y_loc_img, y_loc_img_ignore, y_cls_img, img_meta,
                            label_channels=1, unmap_outputs=True):
        """Compute regression and classification targets for anchors in a
        single image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (N ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (N,).
            y_loc_img (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            img_meta (dict): Meta info of the image.
            y_loc_img_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            img_meta (dict): Meta info of the image.
            y_cls_img (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple:
                y_cls (list[Tensor]): Labels of each level
                label_weights_list (list[Tensor]): Label weights of each level
                y_loc (list[Tensor]): BBox targets of each level
                bbox_weights_list (list[Tensor]): BBox weights of each level
                num_total_pos (int): Number of positive samples in all images
                num_total_neg (int): Number of negative samples in all images
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags, img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        x_i_single = flat_anchors[inside_flags, :]

        assign_result = self.assigner.assign(x_i_single, y_loc_img, y_loc_img_ignore,
                                             None if self.sampling else y_cls_img)
        sampling_result = self.sampler.sample(assign_result, x_i_single, y_loc_img)
        num_valid_anchors = x_i_single.shape[0]
        y_loc_single = torch.zeros_like(x_i_single)
        bbox_weights = torch.zeros_like(x_i_single)
        y_cls_single = x_i_single.new_full((num_valid_anchors, ), self.background_label, dtype=torch.long)
        label_weights = x_i_single.new_zeros(num_valid_anchors, dtype=torch.float)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_y_loc_single = self.bbox_coder.encode(sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_y_loc_single = sampling_result.pos_gt_bboxes
            y_loc_single[pos_inds, :] = pos_y_loc_single
            bbox_weights[pos_inds, :] = 1.0
            if y_cls_img is None:
                # only rpn gives y_cls_img as None, this time FG is 1
                y_cls_single[pos_inds] = 1
            else:
                y_cls_single[pos_inds] = y_cls_img[sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            # fill bg label
            y_cls_single = unmap(y_cls_single, num_total_anchors, inside_flags, fill=self.background_label)
            label_weights = unmap(label_weights, num_total_anchors, inside_flags)
            y_loc_single = unmap(y_loc_single, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
        return y_cls_single, label_weights, y_loc_single, bbox_weights, pos_inds, neg_inds, sampling_result

    def get_targets(self, x_i, valid_flag_list, y_loc_img_list, img_metas, y_loc_img_ignore_list=None,
                    y_cls_img_list=None, label_channels=1, unmap_outputs=True, return_sampling_results=False):
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            x_i (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (N, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (N, )
            y_loc_img_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            y_loc_img_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            y_cls_img_list (list[Tensor]): Ground truth labels of each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.
            return_sampling_results

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - y_cls (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each \
                    level.
                - y_loc (list[Tensor]): BBox targets of each level.
                - bbox_weights_list (list[Tensor]): BBox weights of each level.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end
        """
        num_imgs = len(img_metas)
        assert len(x_i) == len(valid_flag_list) == num_imgs
        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in x_i[0]]
        # concat all level anchors to a single tensor
        concat_x_i = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(x_i[i]) == len(valid_flag_list[i])
            concat_x_i.append(torch.cat(x_i[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))
        # compute targets for each image
        if y_loc_img_ignore_list is None:
            y_loc_img_ignore_list = [None for _ in range(num_imgs)]
        if y_cls_img_list is None:
            y_cls_img_list = [None for _ in range(num_imgs)]
        results = multi_apply(self._get_targets_single, concat_x_i, concat_valid_flag_list,
                              y_loc_img_list, y_loc_img_ignore_list, y_cls_img_list,
                              img_metas, label_channels=label_channels, unmap_outputs=unmap_outputs)
        (all_labels, all_label_weights, all_y_loc_single, all_bbox_weights,
         pos_inds_list, neg_inds_list, sampling_results_list) = results[:7]
        rest_results = list(results[7:])  # user-added return values
        # no valid anchors
        if any([y_cls_single is None for y_cls_single in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        y_cls = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
        y_loc = images_to_levels(all_y_loc_single, num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
        res = (y_cls, label_weights_list, y_loc, bbox_weights_list, num_total_pos, num_total_neg)
        if return_sampling_results:
            res = res + (sampling_results_list, )
        for i, r in enumerate(rest_results):  # user-added return values
            rest_results[i] = images_to_levels(r, num_level_anchors)
        return res + tuple(rest_results)

    def l_det(self, y_head_f_single, y_head_f_r_single, x_i_single, y_cls_single, label_weights,
              y_loc_single, bbox_weights, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            y_head_f_single (Tensor): Box scores for each scale level
                Has shape (n, N * C, H, W).
            y_head_f_r_single (Tensor): Box energies / deltas for each scale
                level with shape (n, N * 4, H, W).
            x_i_single (Tensor): Box reference for each scale level with shape
                (n, num_total_anchors, 4).
            y_cls_single (Tensor): Labels of each anchors with shape
                (n, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (n, num_total_anchors)
            y_loc_single (Tensor): BBox regression targets of each anchor wight
                shape (n, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (n, num_total_anchors, 4).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # classification loss
        y_cls_single = y_cls_single.reshape(-1)
        label_weights = label_weights.reshape(-1)
        y_head_f_single = y_head_f_single.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        l_det_cls = self.FL(y_head_f_single, y_cls_single, label_weights, avg_factor=num_total_samples)
        # regression loss
        y_loc_single = y_loc_single.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        y_head_f_r_single = y_head_f_r_single.permute(0, 2, 3, 1).reshape(-1, 4)
        if self.reg_decoded_bbox:
            x_i_single = x_i_single.reshape(-1, 4)
            y_head_f_r_single = self.bbox_coder.decode(x_i_single, y_head_f_r_single)
        l_det_loc = self.SmoothL1(y_head_f_r_single, y_loc_single, bbox_weights, avg_factor=num_total_samples)
        return l_det_cls, l_det_loc

    # Label Set Training
    @force_fp32(apply_to=('y_f', 'y_f_r'))
    def L_det(self, y_f, y_f_r, y_head_cls, y_loc_img, y_cls_img, img_metas, y_loc_img_ignore=None):
        """Compute losses of the head.

        Args:
            y_f (list[Tensor]): Box scores for each scale level
                Has shape (n, N * C, H, W)
            y_f_r (list[Tensor]): Box energies / deltas for each scale
                level with shape (n, N * 4, H, W)
            y_loc_img (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            y_cls_img (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            y_loc_img_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in y_f]
        assert len(featmap_sizes) == self.anchor_generator.num_levels
        device = y_f[0].device
        x_i, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(x_i, valid_flag_list, y_loc_img, img_metas,
                                           y_loc_img_ignore_list=y_loc_img_ignore,
                                           y_cls_img_list=y_cls_img,
                                           label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (y_cls, label_weights_list, y_loc, bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (num_total_pos + num_total_neg if self.sampling else num_total_pos)
        # anchor number of multi levels
        num_level_anchors = [x_i_single.size(0) for x_i_single in x_i[0]]
        # concat all level anchors and flags to a single tensor
        concat_x_i = []
        for i in range(len(x_i)):
            concat_x_i.append(torch.cat(x_i[i]))
        all_x_i = images_to_levels(concat_x_i, num_level_anchors)
        l_det_cls, l_det_loc = multi_apply(self.l_det, y_f, y_f_r, all_x_i,
                                           y_cls, label_weights_list, y_loc, bbox_weights_list,
                                           num_total_samples=num_total_samples)
        # compute mil loss
        y_head_cls_1level, y_cls_1level = self.get_img_gtlabel_score(y_cls_img, y_head_cls)
        l_imgcls = self.l_imgcls(y_head_cls_1level, y_cls_1level)
        return dict(l_det_cls=l_det_cls, l_det_loc=l_det_loc, l_imgcls=[l_imgcls])

    @force_fp32(apply_to=('y_f', 'y_f_r'))
    def get_img_gtlabel_score(self, y_cls_img, y_head_cls):
        y_head_cls_1level = torch.zeros(len(y_cls_img), self.cls_out_channels).cuda(torch.cuda.current_device())
        y_cls_1level = torch.zeros(len(y_cls_img), self.cls_out_channels).cuda(torch.cuda.current_device())
        for i_img in range(len(y_cls_img)):
            for i_obj in range(len(y_cls_img[i_img])):
                y_cls_1level[i_img, y_cls_img[i_img][i_obj]] = 1
        for y_head_cls_single in y_head_cls:
            y_head_cls_1level = torch.max(y_head_cls_1level, y_head_cls_single.sum(1))
        y_head_cls_1level = y_head_cls_1level.clamp(1e-5, 1.0-1e-5)
        return y_head_cls_1level, y_cls_1level

    def l_wave_dis(self, y_head_f_1_single, y_head_f_2_single, y_head_cls_single, y_head_f_r_single,
                   x_i_single, y_cls_single, label_weights, y_loc_single, bbox_weights, num_total_samples):
        y_head_f_1_single = y_head_f_1_single.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        y_head_f_2_single = y_head_f_2_single.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        y_head_f_1_single = nn.Sigmoid()(y_head_f_1_single)
        y_head_f_2_single = nn.Sigmoid()(y_head_f_2_single)
        # mil weight
        w_i = y_head_cls_single.detach()
        l_det_cls_all = (abs(y_head_f_1_single - y_head_f_2_single) *
                         w_i.reshape(-1, self.cls_out_channels)).mean(dim=1).sum() * self.param_lambda
        l_det_loc = torch.tensor([0.0], device=y_head_f_1_single.device)
        return l_det_cls_all, l_det_loc

    # Re-weighting and minimizing instance uncertainty
    @force_fp32(apply_to=('y_f', 'y_f_r'))
    def L_wave_min(self, y_f, y_f_r, y_head_cls, y_loc_img, y_cls_img, img_metas, y_loc_img_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in y_f[0]]
        assert len(featmap_sizes) == self.anchor_generator.num_levels
        device = y_f[0][0].device
        x_i, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(x_i, valid_flag_list, y_loc_img, img_metas,
                                           y_loc_img_ignore_list=y_loc_img_ignore,
                                           y_cls_img_list=y_cls_img,
                                           label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (y_cls, label_weights_list, y_loc, bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (num_total_pos + num_total_neg if self.sampling else num_total_pos)
        # anchor number of multi levels
        num_level_anchors = [x_i_single.size(0) for x_i_single in x_i[0]]
        # concat all level anchors and flags to a single tensor
        concat_x_i = []
        for i in range(len(x_i)):
            concat_x_i.append(torch.cat(x_i[i]))
        all_x_i = images_to_levels(concat_x_i, num_level_anchors)
        l_wave_dis, l_det_loc = multi_apply(self.l_wave_dis, y_f[0], y_f[1], y_head_cls, y_f_r, all_x_i, y_cls,
                                            label_weights_list, y_loc, bbox_weights_list,
                                            num_total_samples=num_total_samples)
        l_det_cls1, l_det_loc1 = multi_apply(self.l_det, y_f[0], y_f_r, all_x_i,
                                             y_cls, label_weights_list, y_loc, bbox_weights_list,
                                             num_total_samples=num_total_samples)
        l_det_cls2, l_det_loc2 = multi_apply(self.l_det, y_f[1], y_f_r, all_x_i,
                                             y_cls, label_weights_list, y_loc, bbox_weights_list,
                                             num_total_samples=num_total_samples)
        if y_loc_img[0][0][0] < 0:
            l_det_cls = list(map(lambda m, n: (m + n) * 0, l_det_cls1, l_det_cls2))
            l_det_loc = list(map(lambda m, n: (m + n) * 0, l_det_loc1, l_det_loc2))
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
            l_det_cls = list(map(lambda m, n: (m + n) / 2, l_det_cls1, l_det_cls2))
            l_det_loc = list(map(lambda m, n: (m + n) / 2, l_det_loc1, l_det_loc2))
            l_wave_dis = list(map(lambda m: m * 0.0, l_wave_dis))
            # compute mil loss
            y_head_cls_1level, y_cls_1level = self.get_img_gtlabel_score(y_cls_img, y_head_cls)
            l_imgcls = self.l_imgcls(y_head_cls_1level, y_cls_1level)
        return dict(l_det_cls=l_det_cls, l_det_loc=l_det_loc, l_wave_dis=l_wave_dis, l_imgcls=[l_imgcls])

    @force_fp32(apply_to=('y_f', 'y_f_r'))
    def get_img_pseudolabel_score(self, y_f, y_head_cls):
        batch_size = y_head_cls[0].shape[0]
        y_head_cls_1level = torch.zeros(batch_size, self.cls_out_channels).cuda(torch.cuda.current_device())
        y_pseudo = torch.zeros(batch_size, self.cls_out_channels).cuda(torch.cuda.current_device())
        # predict image pseudo label
        with torch.no_grad():
            for s in range(len(y_f[0])):
                y_head_f_i = y_f[0][s].permute(0, 2, 3, 1).reshape(batch_size, -1, self.cls_out_channels).sigmoid()
                y_head_f_i = y_f[1][s].permute(0, 2, 3, 1).reshape(batch_size, -1, self.cls_out_channels).sigmoid() + y_head_f_i
                y_head_f_i = y_head_f_i.max(1)[0] / 2
                y_pseudo = torch.max(y_pseudo, y_head_f_i)
            y_pseudo[y_pseudo >= 0.5] = 1
            y_pseudo[y_pseudo < 0.5] = 0
        # mil image score
        for y_head_cls_single in y_head_cls:
            y_head_cls_1level = torch.max(y_head_cls_1level, y_head_cls_single.sum(1))
        y_head_cls_1level = y_head_cls_1level.clamp(1e-5, 1.0 - 1e-5)
        return y_head_cls_1level, y_pseudo.detach()

    def l_wave_dis_minus(self, y_head_f_1_single, y_head_f_2_single, y_head_cls_single, y_head_f_r_single,
                         x_i_single, y_cls_single, label_weights, y_loc_single, bbox_weights, num_total_samples):
        y_head_f_1_single = y_head_f_1_single.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        y_head_f_2_single = y_head_f_2_single.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        y_head_f_1_single = nn.Sigmoid()(y_head_f_1_single)
        y_head_f_2_single = nn.Sigmoid()(y_head_f_2_single)
        # mil weight
        w_i = y_head_cls_single.detach()
        l_det_cls_all = ((1 - abs(y_head_f_1_single - y_head_f_2_single)) *
                         w_i.view(-1, self.cls_out_channels)).mean(dim=1).sum() * self.param_lambda
        l_det_loc = torch.tensor([0.0], device=y_head_f_1_single.device)
        return l_det_cls_all, l_det_loc

    # Re-weighting and maximizing instance uncertainty
    @force_fp32(apply_to=('y_f', 'y_f_r'))
    def L_wave_max(self, y_f, y_f_r, y_head_cls, y_loc_img, y_cls_img, img_metas, y_loc_img_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in y_f[0]]
        assert len(featmap_sizes) == self.anchor_generator.num_levels
        device = y_f[0][0].device
        x_i, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(x_i, valid_flag_list, y_loc_img, img_metas,
                                           y_loc_img_ignore_list=y_loc_img_ignore,
                                           y_cls_img_list=y_cls_img,
                                           label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (y_cls, label_weights_list, y_loc, bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (num_total_pos + num_total_neg if self.sampling else num_total_pos)
        # anchor number of multi levels
        num_level_anchors = [x_i_single.size(0) for x_i_single in x_i[0]]
        # concat all level anchors and flags to a single tensor
        concat_x_i = []
        for i in range(len(x_i)):
            concat_x_i.append(torch.cat(x_i[i]))
        all_x_i = images_to_levels(concat_x_i, num_level_anchors)
        l_wave_dis_minus, l_det_loc = multi_apply(self.l_wave_dis_minus, y_f[0], y_f[1], y_head_cls, y_f_r, all_x_i,
                                                  y_cls, label_weights_list, y_loc, bbox_weights_list,
                                                  num_total_samples=num_total_samples)
        l_det_cls1, l_det_loc1 = multi_apply(self.l_det, y_f[0], y_f_r, all_x_i,
                                             y_cls, label_weights_list, y_loc, bbox_weights_list,
                                             num_total_samples=num_total_samples)
        l_det_cls2, l_det_loc2 = multi_apply(self.l_det, y_f[1], y_f_r, all_x_i,
                                             y_cls, label_weights_list, y_loc, bbox_weights_list,
                                             num_total_samples=num_total_samples)
        if y_loc_img[0][0][0] < 0:
            l_det_cls = list(map(lambda m, n: (m + n) * 0, l_det_cls1, l_det_cls2))
            l_det_loc = list(map(lambda m, n: (m + n) * 0, l_det_loc1, l_det_loc2))
            for (i, value) in enumerate(l_det_loc):
                if value.isnan():
                    l_det_loc[i].data = torch.tensor(0.0, device=device)
        else:
            l_det_cls = list(map(lambda m, n: (m + n) / 2, l_det_cls1, l_det_cls2))
            l_det_loc = list(map(lambda m, n: (m + n) / 2, l_det_loc1, l_det_loc2))
            l_wave_dis_minus = list(map(lambda m: m * 0.0, l_wave_dis_minus))
        return dict(l_det_cls=l_det_cls, l_det_loc=l_det_loc, l_wave_dis_minus=l_wave_dis_minus)

    @force_fp32(apply_to=('y_f', 'y_f_r'))
    def get_bboxes(self, y_f, y_f_r, img_metas, cfg=None, rescale=False):
        """Transform network output for a batch into bbox predictions.

        Args:
            y_f (list[Tensor]): Box scores for each scale level
                Has shape (n, N * C, H, W)
            y_f_r (list[Tensor]): Box energies / deltas for each scale
                level with shape (n, N * 4, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class labelof the
                corresponding box.

        Example:
            >>> import mmcv
            >>> self = MIAODHead(
            >>>     C=9,
            >>>     in_channels=1,
            >>>     anchor_generator=dict(
            >>>         type='AnchorGenerator',
            >>>         scales=[8],
            >>>         ratios=[0.5, 1.0, 2.0],
            >>>         strides=[4,]))
            >>> img_metas = [{'img_shape': (32, 32, 3), 'scale_factor': 1}]
            >>> cfg = mmcv.Config(dict(
            >>>     score_thr=0.00,
            >>>     nms=dict(type='nms', iou_thr=1.0),
            >>>     max_per_img=10))
            >>> feat = torch.rand(1, 1, 3, 3)
            >>> y_head_f_single, y_head_f_r_single = self.forward_single(feat)
            >>> # note the input lists are over different levels, not images
            >>> y_f, y_f_r = [y_head_f_single], [y_head_f_r_single]
            >>> result_list = self.get_bboxes(y_f, y_f_r,
            >>>                               img_metas, cfg)
            >>> det_bboxes, det_labels = result_list[0]
            >>> assert len(result_list) == 1
            >>> assert det_bboxes.shape[1] == 5
            >>> assert len(det_bboxes) == len(det_labels) == cfg.max_per_img
        """
        assert len(y_f) == len(y_f_r)
        num_levels = len(y_f)
        device = y_f[0].device
        featmap_sizes = [y_f[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(featmap_sizes, device=device)
        result_list = []
        for img_id in range(len(img_metas)):
            y_head_f_single_list = [y_f[i][img_id].detach() for i in range(num_levels)]
            y_head_f_r_single_list = [y_f_r[i][img_id].detach() for i in range(num_levels)]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(y_head_f_single_list, y_head_f_r_single_list,
                                                mlvl_anchors, img_shape, scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self, y_head_f_single_list, y_head_f_r_single_list,
                           mlvl_anchors, img_shape, scale_factor, cfg, rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            y_head_f_single_list (list[Tensor]): Box scores for a single scale level
                Has shape (N * C, H, W).
            y_head_f_r_single_list (list[Tensor]): Box energies / deltas for a single
                scale level with shape (N * 4, H, W).
            mlvl_anchors (list[Tensor]): Box reference for a single scale level
                with shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(y_head_f_single_list) == len(y_head_f_r_single_list) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for y_head_f_single, y_head_f_r_single, x_i_single in zip(y_head_f_single_list,
                                                                  y_head_f_r_single_list, mlvl_anchors):
            assert y_head_f_single.size()[-2:] == y_head_f_r_single.size()[-2:]
            y_head_f_single = y_head_f_single.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = y_head_f_single.sigmoid()
            else:
                scores = y_head_f_single.softmax(-1)
            # scores = y_head_f_single
            y_head_f_r_single = y_head_f_r_single.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if 0 < nms_pre < scores.shape[0]:
                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    # remind that we set FG labels to [0, num_class-1]
                    # since mmdet v2.0
                    # BG cat_id: num_class
                    max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                x_i_single = x_i_single[topk_inds, :]
                y_head_f_r_single = y_head_f_r_single[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = self.bbox_coder.decode(x_i_single, y_head_f_r_single, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            # Add a dummy background class to the backend when using sigmoid
            # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
            # BG cat_id: num_class
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores, cfg.score_thr, cfg.nms, cfg.max_per_img)
        return det_bboxes, det_labels

    def loss(self, **kwargs):
        # This function is to avoid the TypeError caused by the abstract method defined in "base_dense_head.py".
        return
