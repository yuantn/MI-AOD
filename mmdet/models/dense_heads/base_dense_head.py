from abc import ABCMeta, abstractmethod

import torch.nn as nn
from tools.utils import losstype


class BaseDenseHead(nn.Module, metaclass=ABCMeta):
    """Base class for DenseHeads."""

    def __init__(self):
        super(BaseDenseHead, self).__init__()

    @abstractmethod
    def loss(self, **kwargs):
        """Compute losses of the head."""
        pass

    @abstractmethod
    def get_bboxes(self, **kwargs):
        """Transform network output for a batch into bbox predictions."""
        pass

    def forward_train(self,
                      x,
                      img_metas,
                      y_loc_img,
                      y_cls_img=None,
                      y_loc_img_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            y_loc_img (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            y_cls_img (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            y_loc_img_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        y_head_f_1, y_head_f_2, y_head_f_r, y_head_cls = self(x)
        # Label set training
        if losstype.losstype == 0:
            outs = (y_head_f_1, y_head_f_r, y_head_cls)
            if y_cls_img is None:
                loss_inputs = outs + (y_loc_img, img_metas)
            else:
                loss_inputs = outs + (y_loc_img, y_cls_img, img_metas)
            L_det_1 = self.L_det(*loss_inputs, y_loc_img_ignore=y_loc_img_ignore)
            outs = (y_head_f_2, y_head_f_r, y_head_cls)
            if y_cls_img is None:
                loss_inputs = outs + (y_loc_img, img_metas)
            else:
                loss_inputs = outs + (y_loc_img, y_cls_img, img_metas)
            L_det_2 = self.L_det(*loss_inputs, y_loc_img_ignore=y_loc_img_ignore)
            l_det_cls = list(map(lambda m, n: (m + n)/2, L_det_1['l_det_cls'], L_det_2['l_det_cls']))
            l_det_loc = list(map(lambda m, n: (m + n)/2, L_det_1['l_det_loc'], L_det_2['l_det_loc']))
            l_imgcls = list(map(lambda m, n: (m + n)/2, L_det_1['l_imgcls'], L_det_2['l_imgcls']))
            L_det = dict(l_det_cls=l_det_cls, l_det_loc=l_det_loc, l_imgcls=l_imgcls)
            if proposal_cfg is None:
                return L_det
            else:
                proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
                return L_det, proposal_list
        # Re-weighting and minimizing instance uncertainty
        elif losstype.losstype == 1:
            outs = ((y_head_f_1, y_head_f_2), y_head_f_r, y_head_cls)
            if y_cls_img is None:
                loss_inputs = outs + (y_loc_img, img_metas)
            else:
                loss_inputs = outs + (y_loc_img, y_cls_img, img_metas)
            loss = self.L_wave_min(*loss_inputs, y_loc_img_ignore=y_loc_img_ignore)
            L_wave_min = dict(l_det_cls=loss['l_det_cls'], l_det_loc=loss['l_det_loc'],
                              l_wave_dis=loss['l_wave_dis'], l_imgcls=loss['l_imgcls'])
            if proposal_cfg is None:
                return L_wave_min
            else:
                proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
                return L_wave_min, proposal_list
        # Re-weighting and maximizing instance uncertainty
        else:
            outs = ((y_head_f_1, y_head_f_2), y_head_f_r, y_head_cls)
            if y_cls_img is None:
                loss_inputs = outs + (y_loc_img, img_metas)
            else:
                loss_inputs = outs + (y_loc_img, y_cls_img, img_metas)
            loss = self.L_wave_max(*loss_inputs, y_loc_img_ignore=y_loc_img_ignore)
            L_wave_max = dict(l_det_cls=loss['l_det_cls'], l_det_loc=loss['l_det_loc'],
                              l_wave_dis_minus=loss['l_wave_dis_minus'])
            if proposal_cfg is None:
                return L_wave_max
            else:
                proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
                return L_wave_max, proposal_list
