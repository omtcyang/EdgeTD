import torch
import torch.nn as nn

BatchNorm2d = nn.BatchNorm2d
import torch.nn.functional as F

import math
import numpy as np
import cv2
import time
import copy
from PIL import Image
import os
import os.path as osp
from shapely.geometry import Polygon
import pyclipper
from copy import deepcopy as cdc

from torchvision.ops.misc import Conv2dNormActivation


class DiceLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(DiceLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, input, target, mask, reduce=True):
        batch_size = input.size(0)
        input = torch.sigmoid(input)

        input = input.contiguous().view(batch_size, -1)
        target = target.contiguous().view(batch_size, -1).float()
        mask = mask.contiguous().view(batch_size, -1).float()

        input = input * mask
        target = target * mask

        a = torch.sum(input * target, dim=1)
        b = torch.sum(input * input, dim=1) + 0.001
        c = torch.sum(target * target, dim=1) + 0.001
        d = (2 * a) / (b + c)
        dice_loss = 1 - d

        dice_loss = self.loss_weight * dice_loss

        if reduce:
            dice_loss = torch.mean(dice_loss)

        return dice_loss


# class ParamLoss(nn.Module):
#     def __init__(self, loss_weight=1.0):
#         super(ParamLoss, self).__init__()
#         self.loss_weight = loss_weight

#     def forward(self, gt_param, pred_param, x_ranges=None, reduce=True):
#         # EuclideanDistance2ParamLoss
#         if x_ranges is not None: 
#             x_range, x2_range = x_ranges
#             diff_param = (gt_param - pred_param)

#             up_diff = diff_param[:, 0][:, None] * x2_range[None, :]
#             down_diff = diff_param[:, 1][:, None] * x2_range[None, :]

#             # up_diff = torch.abs(up_diff)
#             # down_diff = torch.abs(down_diff)
#             # loss_param = (torch.sum(up_diff) + torch.sum(down_diff)) / up_diff.shape[0]

#             curve_diff = torch.abs(up_diff) + torch.abs(down_diff)
#             curve_diff = torch.sum(curve_diff, dim=-1)/up_diff.shape[1]

#             loss_param = torch.sum(curve_diff) / up_diff.shape[0]

#         return loss_param


class ParamLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(ParamLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, gt_param, pred_param, x_ranges=None, reduce=True):
        # EuclideanDistance2ParamLoss
        if x_ranges is not None: 
            x_range, x2_range = x_ranges
            diff_param = (gt_param - pred_param)

            # 上边
            up_gt = (gt_param[:, 0][:, None] * x2_range[None, :])[:,:,None]
            up_pred = (pred_param[:, 0][:, None] * x2_range[None, :])[:,:,None]


            up_gt_pred_union = torch.max(torch.abs(torch.cat([up_gt, up_pred], dim=-1)),dim=-1)
            up_gt_pred_union_integral = torch.sum(up_gt_pred_max, dim=-1)

            up_diff = torch.abs(diff_param[:, 0][:, None] * x2_range[None, :])
            up_diff_integral = torch.sum(up_diff, dim=-1)

            up_loss_DiffOU = up_diff_integral/up_gt_pred_union_integral
            up_loss_DiffOU[up_loss_DiffOU>1]=1
            # up_gt_pred_direction = gt_param[:, 0]*pred_param[:, 0]
            # index_reverse = up_gt_pred_direction<0
            # up_loss_DiffOU[index_reverse]=1

            # 下边
            down_gt = (gt_param[:, 1][:, None] * x2_range[None, :])[:,:,None]
            down_pred = (pred_param[:, 1][:, None] * x2_range[None, :])[:,:,None]

            down_gt_pred_union = torch.max(torch.abs(torch.cat([down_gt, down_pred], dim=-1)),dim=-1)
            down_gt_pred_union_integral = torch.sum(down_gt_pred_max, dim=-1)

            down_diff = torch.abs(diff_param[:, 1][:, None] * x2_range[None, :])
            down_diff_integral = torch.sum(down_diff, dim=-1)

            down_loss_DiffOU = down_diff_integral/down_gt_pred_union_integral
            down_loss_DiffOU[down_loss_DiffOU>1]=1

            loss_DiffOU = torch.sum(np.sqrt(up_loss_DiffOU)/2 + np.sqrt(down_loss_DiffOU)/2)/up_diff.shape[1]

        return loss_DiffOU


class ShiftLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(ShiftLoss, self).__init__()
        self.loss_weight = loss_weight


    def decode_shift2box(feature_map):
        height, width = feature_map.shape[1:3]

        # 构造偏移量矩阵
        offsets = feature_map.transpose(1, 2, 0).reshape(height, width, -1, 2)

        # 计算四个角点的坐标
        x_coords = np.arange(width)
        y_coords = np.arange(height)
        xx, yy = np.meshgrid(x_coords, y_coords)
        pixel_coords = np.stack([xx, yy], axis=-1)
        corner_coords = pixel_coords[:, :, np.newaxis, :] + offsets

        # 从四个角点的坐标中构造 box
        x1 = corner_coords[:, :, 0, 0]
        y1 = corner_coords[:, :, 0, 1]
        x2 = corner_coords[:, :, 1, 0]
        y2 = corner_coords[:, :, 1, 1]
        x3 = corner_coords[:, :, 2, 0]
        y3 = corner_coords[:, :, 2, 1]
        x4 = corner_coords[:, :, 3, 0]
        y4 = corner_coords[:, :, 3, 1]

        decoded_boxes = np.stack([x1, y1, x2, y2, x3, y3, x4, y4], axis=-1)
        return decoded_boxes


    def calculate_iou(box1, box2):
        # 将四个角点坐标拆分
        x1_tl, y1_tl = box1[:, 0], box1[:, 1]
        x1_tr, y1_tr = box1[:, 2], box1[:, 3]
        x1_br, y1_br = box1[:, 4], box1[:, 5]
        x1_bl, y1_bl = box1[:, 6], box1[:, 7]

        x2_tl, y2_tl = box2[:, 0], box2[:, 1]
        x2_tr, y2_tr = box2[:, 2], box2[:, 3]
        x2_br, y2_br = box2[:, 4], box2[:, 5]
        x2_bl, y2_bl = box2[:, 6], box2[:, 7]

        # 计算交集的坐标
        x_intersection_tl = torch.max(x1_tl, x2_tl)
        y_intersection_tl = torch.max(y1_tl, y2_tl)
        x_intersection_br = torch.min(x1_br, x2_br)
        y_intersection_br = torch.min(y1_br, y2_br)

        # 计算交集的面积
        intersection_area = torch.clamp(x_intersection_br - x_intersection_tl, min=0) * torch.clamp(y_intersection_br - y_intersection_tl, min=0)

        # 计算并集的面积
        area1 = (x1_br - x1_tl) * (y1_br - y1_tl)
        area2 = (x2_br - x2_tl) * (y2_br - y2_tl)
        union_area = area1 + area2 - intersection_area

        # 计算 GIOU
        iou = intersection_area / union_area
        xA = torch.min(x1_tl, x2_tl)
        yA = torch.min(y1_tl, y2_tl)
        xB = torch.max(x1_br, x2_br)
        yB = torch.max(y1_br, y2_br)
        inter_area = torch.clamp((xB - xA), min=0) * torch.clamp((yB - yA), min=0)
        comp = inter_area / union_area
        giou = iou - (comp - (1 - torch.abs(comp)))

        # 计算角度差异
        atan1 = torch.atan2(x1_tl + x1_br - x2_tl - x2_br, y1_tl + y1_br - y2_tl - y2_br)
        atan2 = torch.atan2(x1_br - x1_tr - x2_br + x2_tr, y1_br - y1_tr - y2_br + y2_tr)
        angle_diff = atan1 - atan2

        # 计算 CIOU
        v = (4 / (math.pi ** 2)) * (angle_diff - torch.pow((2 * math.atan(angle_diff)), 2))
        ciou = giou - v

        return ciou


    def forward(self, gt_shift, pred_shift, gt_shrink4filter, reduce=True):
        # 示例输入
        # feature_map = np.random.randn(8, 10, 10)  # 假设feature_map是一个8通道的特征图，大小为10x10

        # 解码特征图
        pred_boxes = decode_shift2box(pred_shift)[gt_shrink4filter.reshape(-1)]
        gt_boxes = decode_shift2box(gt_shift)[gt_shrink4filter.reshape(-1)]
        loss_ciou = calculate_iou(pred_boxes, gt_boxes)

        return loss_param