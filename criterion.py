import torch
import torch.nn as nn

import torch.nn.functional as F

from .losses import DiceLoss, ParamLoss


class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()

        # self.criterion_shift = ShiftLoss()
        self.criterion_param = ParamLoss()
        # criterion_centroid = F.binary_cross_entropy_with_logits()
        self.criterion_shrink = DiceLoss()

    def forward(self, preds, gt_labels, x_ranges):
        shrink, centroid, param, shift = preds
        gt_valid_mask, gt_shrink, gt_centroid, gt_param, gt_shift = gt_labels
        # print('preds',gt_valid_mask.shape, shrink.shape, centroid.shape, param.shape, shift.shape)
        # print('gt',gt_valid_mask.shape, gt_shrink.shape, gt_centroid.shape, gt_param.shape, gt_shift.shape)

        gt_shrink4filter = gt_shrink.bool()

        # shift loss
        # pred_shift = shift.permute(0, 2, 3, 1).reshape(-1, 8)[gt_shrink4filter]
        # gt_shift = gt_shift.permute(0, 2, 3, 1).reshape(-1, 8)[gt_shrink4filter]  
        loss_shift = self.criterion_shift(gt_shift, pred_shift, gt_shrink4filter)
        # loss_shift = torch.mean(torch.sum(F.smooth_l1_loss(pred_shift, gt_shift, reduction='none'), dim=-1))

        # param loss
        pred_param = param.permute(0, 2, 3, 1).reshape(-1, 2)[gt_shrink4filter.reshape(-1)]
        gt_param = gt_param.permute(0, 2, 3, 1).reshape(-1, 2)[gt_shrink4filter.reshape(-1)]  
        loss_param = self.criterion_param(gt_param, pred_param, x_ranges)

        # centroid loss
        pred_centroid = (centroid.permute(0, 2, 3, 1).reshape(-1)[gt_shrink4filter.reshape(-1)]).sigmoid()
        gt_centroid = gt_centroid.reshape(-1)[gt_shrink4filter.reshape(-1)]
        loss_centroid = F.binary_cross_entropy_with_logits(pred_centroid, gt_centroid)

        # shrink
        pred_shrink = (shrink[:, 0, :, :]).sigmoid()
        gt_shrink = gt_shrink[:, 0, :, :]
        gt_valid_mask = gt_valid_mask[:, 0, :, :]
        loss_shrink=self.criterion_shrink(pred_shrink, gt_shrink, gt_valid_mask)

        return loss_shift, loss_param, loss_centroid, loss_shrink


def build_criterion(args):
    return Criterion()