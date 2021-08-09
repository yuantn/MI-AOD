import torch
import torch.nn as nn

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector


@DETECTORS.register_module()
class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self, backbone, neck=None, bbox_head=None, train_cfg=None, test_cfg=None,
                 pretrained=None):
        super(SingleStageDetector, self).__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(SingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, x):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, x):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        x = self.extract_feat(x)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self, x, img_metas, y_loc_img, y_cls_img, y_loc_img_ignore=None):
        """
        Args:
            x (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            y_loc_img (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            y_cls_img (list[Tensor]): Class indices corresponding to each box
            y_loc_img_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x = self.extract_feat(x)
        losses = self.bbox_head.forward_train(x, img_metas, y_loc_img, y_cls_img, y_loc_img_ignore)
        return losses

    def simple_test(self, x, img_metas, return_box=True, rescale=False):
        """Test function without test time augmentation.

        Args:
            xs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            np.ndarray: proposals
        """
        x = self.extract_feat(x)
        y_head_f_1, y_head_f_2, y_head_f_r, y_head_cls = self.bbox_head(x)
        if not return_box:
            y_head_f_1_1level = []
            y_head_f_2_1level = []
            for y_head_f_i_single in y_head_f_1:
                y_head_f_1_1level.append(y_head_f_i_single.permute(0,2,3,1).reshape(-1, self.bbox_head.cls_out_channels))
            for y_head_f_i_single in y_head_f_2:
                y_head_f_2_1level.append(y_head_f_i_single.permute(0,2,3,1).reshape(-1, self.bbox_head.cls_out_channels))
            return y_head_f_1_1level, y_head_f_2_1level, y_head_cls
        outs = (y_head_f_1, y_head_f_r)
        y_head_loc_cls = self.bbox_head.get_bboxes(*outs, img_metas, rescale=rescale)
        # skip post-processing when exporting to ONNX
        if torch.onnx.is_in_onnx_export():
            return y_head_loc_cls
        y_head = [bbox2result(y_head_loc, y_head_cls, self.bbox_head.C)
                        for y_head_loc, y_head_cls in y_head_loc_cls]
        return y_head[0]

    def aug_test(self, x, img_metas, rescale=False):
        """Test function with test time augmentation."""
        raise NotImplementedError
