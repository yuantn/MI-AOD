import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import (build_anchor_generator, build_assigner,
                        build_bbox_coder, build_sampler, multi_apply)
from ..builder import HEADS
from ..losses import smooth_l1_loss
from .anchor_head import AnchorHead

class MyEntLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # p = torch.nn.functional.softmax(x, dim=1)
        # x = torch.reshape(x, (x.shape[0]*x.shape[1],1)).squeeze(-1)
        x = torch.nn.Softmax(dim=1)(x)
        p = x / torch.repeat_interleave(x.sum(dim=1).unsqueeze(-1), repeats=21, dim=1)
        logp = torch.log2(p)
        ent = -torch.mul(p, logp)
        entloss = torch.sum(ent, dim=1)
        # entloss = torch.mean(entloss) + torch.tensor(0.01).cuda()
        return entloss

class Mycosloss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        # return torch.tensor(1).cuda() - torch.mean(torch.cosine_similarity(x1, x2)).cuda()
        return torch.tensor(1).cuda() - torch.cosine_similarity(x1, x2).cuda()

class Myrecosloss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        # return torch.mean(torch.cosine_similarity(x1, x2)).cuda() + torch.tensor(1).cuda()
        return torch.cosine_similarity(x1, x2).cuda() + torch.tensor(1).cuda()


# TODO: add loss evaluator for SSD
@HEADS.register_module()
class SSDHead(AnchorHead):
    """SSD head used in https://arxiv.org/abs/1512.02325.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        anchor_generator (dict): Config dict for anchor generator
        background_label (int | None): Label ID of background, set as 0 for
            RPN and num_classes for other heads. It will automatically set as
            num_classes if None is given.
        bbox_coder (dict): Config of bounding box coder.
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied on decoded bounding boxes. Default: False
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
    """  # noqa: W605

    def __init__(self,
                 num_classes=80,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 anchor_generator=dict(
                     type='SSDAnchorGenerator',
                     scale_major=False,
                     input_size=300,
                     strides=[8, 16, 32, 64, 100, 300],
                     ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2]),
                     basesize_ratio_range=(0.1, 0.9)),
                 background_label=None,
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     target_means=[.0, .0, .0, .0],
                     target_stds=[1.0, 1.0, 1.0, 1.0],
                 ),
                 reg_decoded_bbox=False,
                 train_cfg=None,
                 test_cfg=None):
        super(AnchorHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.cls_out_channels = num_classes + 1  # add background class
        self.anchor_generator = build_anchor_generator(anchor_generator)
        num_anchors = self.anchor_generator.num_base_anchors

        reg_convs = []
        cls_convs1 = []
        cls_convs2 = []
        for i in range(len(in_channels)):
            reg_convs.append(
                nn.Conv2d(
                    in_channels[i],
                    num_anchors[i] * 4,
                    kernel_size=3,
                    padding=1))
            cls_convs1.append(
                nn.Conv2d(
                    in_channels[i],
                    num_anchors[i] * (num_classes + 1),
                    kernel_size=3,
                    padding=1
                ))
            cls_convs2.append(
                nn.Conv2d(
                    in_channels[i],
                    num_anchors[i] * (num_classes + 1),
                    kernel_size=3,
                    padding=1
                ))
        self.reg_convs = nn.ModuleList(reg_convs)
        self.cls_convs1 = nn.ModuleList(cls_convs1)
        self.cls_convs2 = nn.ModuleList(cls_convs2)

        self.background_label = (
            num_classes if background_label is None else background_label)
        # background_label should be either 0 or num_classes
        assert (self.background_label == 0
                or self.background_label == num_classes)

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

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Classification scores for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * 4.
        """

        #  add another cls
        cls_scores1 = []
        cls_scores2 = []
        bbox_preds = []
        for feat, reg_conv, cls_conv1, cls_conv2 in zip(feats, self.reg_convs,
                                            self.cls_convs1, self.cls_convs2):
            cls_scores1.append(cls_conv1(feat))
            cls_scores2.append(cls_conv2(feat))
            bbox_preds.append(reg_conv(feat))
        return cls_scores1, cls_scores2, bbox_preds

    def loss_single(self, cls_score, bbox_pred, anchor, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        """Compute loss of a single image.

        Args:
            cls_score (Tensor): Box scores for eachimage
                Has shape (num_total_anchors, num_classes).
            bbox_pred (Tensor): Box energies / deltas for each image
                level with shape (num_total_anchors, 4).
            anchors (Tensor): Box reference for each scale level with shape
                (num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (num_total_anchors,).
            label_weights (Tensor): Label weights of each anchor with shape
                (num_total_anchors,)
            bbox_targets (Tensor): BBox regression targets of each anchor wight
                shape (num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (num_total_anchors, 4).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        loss_cls_all = F.cross_entropy(
            cls_score, labels, reduction='none') * label_weights
        # calculate my loss

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        pos_inds = ((labels >= 0) &
                    (labels < self.background_label)).nonzero().reshape(-1)
        neg_inds = (labels == self.background_label).nonzero().view(-1)
        # fore/background partition

        if pos_inds.dim() == 0:
            return loss_cls_all.sum()[None]*0, loss_cls_all.sum()

        num_pos_samples = pos_inds.size(0)
        num_neg_samples = self.train_cfg.neg_pos_ratio * num_pos_samples
        if num_neg_samples > neg_inds.size(0):
            num_neg_samples = neg_inds.size(0)
        topk_loss_cls_neg, _ = loss_cls_all[neg_inds].topk(num_neg_samples)
        loss_cls_pos = loss_cls_all[pos_inds].sum()
        loss_cls_neg = topk_loss_cls_neg.sum()
        # loss for pos and neg (our loss only use pos)
        # need 2 loss input (create a new function: loss_minmax and loss_minmax_single)

        loss_cls = (loss_cls_pos + loss_cls_neg) / num_total_samples

        if self.reg_decoded_bbox:
            bbox_pred = self.bbox_coder.decode(anchor, bbox_pred)

        loss_bbox = smooth_l1_loss(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            beta=self.train_cfg.smoothl1_beta,
            avg_factor=num_total_samples)
        return loss_cls[None], loss_bbox

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=1,
            unmap_outputs=False)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        num_images = len(img_metas)
        all_cls_scores = torch.cat([
            s.permute(0, 2, 3, 1).reshape(
                num_images, -1, self.cls_out_channels) for s in cls_scores
        ], 1)
        all_labels = torch.cat(labels_list, -1).view(num_images, -1)
        all_label_weights = torch.cat(label_weights_list,
                                      -1).view(num_images, -1)
        all_bbox_preds = torch.cat([
            b.permute(0, 2, 3, 1).reshape(num_images, -1, 4)
            for b in bbox_preds
        ], -2)
        all_bbox_targets = torch.cat(bbox_targets_list,
                                     -2).view(num_images, -1, 4)
        all_bbox_weights = torch.cat(bbox_weights_list,
                                     -2).view(num_images, -1, 4)

        # concat all level anchors to a single tensor
        all_anchors = []
        for i in range(num_images):
            all_anchors.append(torch.cat(anchor_list[i]))

        # check NaN and Inf
        assert torch.isfinite(all_cls_scores).all().item(), \
            'classification scores become infinite or NaN!'
        assert torch.isfinite(all_bbox_preds).all().item(), \
            'bbox predications become infinite or NaN!'

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            all_cls_scores,
            all_bbox_preds,
            all_anchors,
            all_labels,
            all_label_weights,
            all_bbox_targets,
            all_bbox_weights,
            num_total_samples=num_total_pos)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

#################### loss-single agreement ##################################
    def loss_single_min(self, cls_score, cls_score2, bbox_pred, anchor, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        """Compute loss of a single image.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # loss_cls_all = F.cross_entropy(
        #     cls_score, labels, reduction='none') * label_weights
        ent_list = torch.tensor([]).cuda()

        criterion = Mycosloss()
        #softmax
        min_discrepantloss_un = criterion(cls_score, cls_score2) * label_weights

        ent1_un = MyEntLoss().forward(cls_score)
        ent2_un = MyEntLoss().forward(cls_score2)

        ent_un_cat = torch.cat((torch.unsqueeze(ent1_un, 1), torch.unsqueeze(ent2_un, 1)), dim=1)
        best_ent_un = torch.mean(ent_un_cat, dim=1)
        ent_list = torch.cat((ent_list, best_ent_un.detach()))
        ent_un_transf = nn.Softmax(dim=0)(best_ent_un - 0.1)
        min_disc_wt = (1 - ent_un_transf)
        # 128
        min_disc_wt_avg = (min_disc_wt / 28).detach()
        loss_cls_all = torch.mul(min_disc_wt_avg, min_discrepantloss_un)
        # calculate my loss MyEntLoss

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        pos_inds = ((labels >= 0) &
                    (labels < self.background_label)).nonzero().reshape(-1)
        neg_inds = (labels == self.background_label).nonzero().view(-1)
        # fore/background partition

        num_pos_samples = pos_inds.size(0)
        if num_pos_samples == 0:
            loss_cls = loss_cls_all.sum() * 0
            loss_bbox = loss_cls_all.sum() * 0
        else:
            num_neg_samples = self.train_cfg.neg_pos_ratio * num_pos_samples
            if num_neg_samples > neg_inds.size(0):
                num_neg_samples = neg_inds.size(0)
            # topk_loss_cls_neg, _ = loss_cls_all[neg_inds].topk(num_neg_samples)
            loss_cls_pos = loss_cls_all[pos_inds].sum()
            # loss_cls_neg = topk_loss_cls_neg.sum()
            # loss for pos and neg (our loss only use pos)
            # need 2 loss input (create a new function: loss_minmax and loss_minmax_single)

            # loss_cls = (loss_cls_pos + loss_cls_neg) / num_total_samples
            loss_cls = loss_cls_pos / num_pos_samples

            if self.reg_decoded_bbox:
                bbox_pred = self.bbox_coder.decode(anchor, bbox_pred)

            loss_bbox = smooth_l1_loss(
                bbox_pred,
                bbox_targets,
                bbox_weights,
                beta=self.train_cfg.smoothl1_beta,
                avg_factor=num_total_samples)
        return loss_cls[None], loss_bbox

################################## loss agreement ####################################
    def loss_min(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores[0]]
        # featmap_sizes2 = [featmap.size()[-2:] for featmap in cls_scores[1]]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0][0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=1,
            unmap_outputs=False)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        num_images = len(img_metas)
        all_cls_scores_1 = torch.cat([
            s.permute(0, 2, 3, 1).reshape(
                num_images, -1, self.cls_out_channels) for s in cls_scores[0]
        ], 1)

        all_cls_scores_2 = torch.cat([
            s.permute(0, 2, 3, 1).reshape(
                num_images, -1, self.cls_out_channels) for s in cls_scores[1]
        ], 1)

        all_labels = torch.cat(labels_list, -1).view(num_images, -1)
        all_label_weights = torch.cat(label_weights_list,
                                      -1).view(num_images, -1)
        all_bbox_preds = torch.cat([
            b.permute(0, 2, 3, 1).reshape(num_images, -1, 4)
            for b in bbox_preds
        ], -2)
        all_bbox_targets = torch.cat(bbox_targets_list,
                                     -2).view(num_images, -1, 4)
        all_bbox_weights = torch.cat(bbox_weights_list,
                                     -2).view(num_images, -1, 4)

        # concat all level anchors to a single tensor
        all_anchors = []
        for i in range(num_images):
            all_anchors.append(torch.cat(anchor_list[i]))

        # check NaN and Inf
        # assert torch.isfinite(all_cls_scores).all().item(), \
        #     'classification scores become infinite or NaN!'
        # assert torch.isfinite(all_bbox_preds).all().item(), \
        #     'bbox predications become infinite or NaN!'

        losses_cls, losses_bbox = multi_apply(
            self.loss_single_min,
            all_cls_scores_1,
            all_cls_scores_2,
            all_bbox_preds,
            all_anchors,
            all_labels,
            all_label_weights,
            all_bbox_targets,
            all_bbox_weights,
            num_total_samples=num_total_pos)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

#################### loss-single discrepancy ##################################
    def loss_single_max(self, cls_score, cls_score2, bbox_pred, anchor, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        """Compute loss of a single image.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # loss_cls_all = F.cross_entropy(
        #     cls_score, labels, reduction='none') * label_weights
        ent_list = torch.tensor([]).cuda()

        criterion = Myrecosloss()
        max_discrepantloss_un = criterion(cls_score, cls_score2) * label_weights

        # baseline need change class: mycosloss
        ent1_un = MyEntLoss().forward(cls_score)
        ent2_un = MyEntLoss().forward(cls_score2)

        ent_un_cat = torch.cat((torch.unsqueeze(ent1_un, 1), torch.unsqueeze(ent2_un, 1)), dim=1)
        best_ent_un = torch.mean(ent_un_cat, dim=1)
        # ent_list = torch.cat((ent_list, best_ent_un.detach()))
        ent_un_transf = nn.Softmax(dim=0)(best_ent_un - 0.1)
        max_disc_wt = ent_un_transf
        max_disc_wt_avg = (max_disc_wt / 28).detach()
        loss_cls_all = torch.mul(max_disc_wt_avg, max_discrepantloss_un)
        # calculate my loss MyEntLoss

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        pos_inds = ((labels >= 0) &
                    (labels < self.background_label)).nonzero().reshape(-1)
        neg_inds = (labels == self.background_label).nonzero().view(-1)
        # fore/background partition

        num_pos_samples = pos_inds.size(0)
        if num_pos_samples == 0:
            loss_cls = loss_cls_all.sum() * 0
            loss_bbox = loss_cls_all.sum() * 0
        else:
            num_neg_samples = self.train_cfg.neg_pos_ratio * num_pos_samples
            if num_neg_samples > neg_inds.size(0):
                num_neg_samples = neg_inds.size(0)
            # topk_loss_cls_neg, _ = loss_cls_all[neg_inds].topk(num_neg_samples)
            loss_cls_pos = loss_cls_all[pos_inds].sum()
            # loss_cls_neg = topk_loss_cls_neg.sum()
            # loss for pos and neg (our loss only use pos)
            # need 2 loss input (create a new function: loss_minmax and loss_minmax_single)

            # loss_cls = (loss_cls_pos + loss_cls_neg) / num_total_samples
            loss_cls = loss_cls_pos / num_pos_samples

            if self.reg_decoded_bbox:
                bbox_pred = self.bbox_coder.decode(anchor, bbox_pred)

            loss_bbox = smooth_l1_loss(
                bbox_pred,
                bbox_targets,
                bbox_weights,
                beta=self.train_cfg.smoothl1_beta,
                avg_factor=num_total_samples)
        return loss_cls[None], loss_bbox

################################## loss discrepancy ####################################
    def loss_max(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores[0]]
        # featmap_sizes2 = [featmap.size()[-2:] for featmap in cls_scores[1]]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0][0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=1,
            unmap_outputs=False)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        num_images = len(img_metas)
        all_cls_scores_1 = torch.cat([
            s.permute(0, 2, 3, 1).reshape(
                num_images, -1, self.cls_out_channels) for s in cls_scores[0]
        ], 1)

        all_cls_scores_2 = torch.cat([
            s.permute(0, 2, 3, 1).reshape(
                num_images, -1, self.cls_out_channels) for s in cls_scores[1]
        ], 1)

        all_labels = torch.cat(labels_list, -1).view(num_images, -1)
        all_label_weights = torch.cat(label_weights_list,
                                      -1).view(num_images, -1)
        all_bbox_preds = torch.cat([
            b.permute(0, 2, 3, 1).reshape(num_images, -1, 4)
            for b in bbox_preds
        ], -2)
        all_bbox_targets = torch.cat(bbox_targets_list,
                                     -2).view(num_images, -1, 4)
        all_bbox_weights = torch.cat(bbox_weights_list,
                                     -2).view(num_images, -1, 4)

        # concat all level anchors to a single tensor
        all_anchors = []
        for i in range(num_images):
            all_anchors.append(torch.cat(anchor_list[i]))

        # check NaN and Inf
        # assert torch.isfinite(all_cls_scores).all().item(), \
        #     'classification scores become infinite or NaN!'
        # assert torch.isfinite(all_bbox_preds).all().item(), \
        #     'bbox predications become infinite or NaN!'

        losses_cls, losses_bbox = multi_apply(
            self.loss_single_max,
            all_cls_scores_1,
            all_cls_scores_2,
            all_bbox_preds,
            all_anchors,
            all_labels,
            all_label_weights,
            all_bbox_targets,
            all_bbox_weights,
            num_total_samples=num_total_pos)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)