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


class Decoder(nn.Module):
    def __init__(self,
                 in_channels=None):
        super(Decoder, self).__init__()

        assert in_channels!=None

        self.sc_smooth_layer = nn.Sequential(
            Conv2dNormActivation(
                in_channels, in_channels, kernel_size=3, padding=1, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU
            ),
            Conv2dNormActivation(
                in_channels, in_channels//2, kernel_size=3, padding=1, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU
            ))

        self.shrink_head = nn.Sequential(
            nn.Conv2d(in_channels//2, 1, 3, padding=1))

        self.centroid_head = nn.Sequential(
            nn.Conv2d(in_channels//2, 1, 3, padding=1))


        self.param_head = nn.Sequential(
            Conv2dNormActivation(
                in_channels, in_channels, kernel_size=3, padding=1, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU
            ),
            Conv2dNormActivation(
                in_channels, in_channels//2, kernel_size=3, padding=1, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU
            ),
            nn.Conv2d(in_channels//2, 2, 3, padding=1))

        self.shift_head = nn.Sequential(
            Conv2dNormActivation(
                in_channels, in_channels, kernel_size=3, padding=1, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU
            ),
            Conv2dNormActivation(
                in_channels, in_channels//2, kernel_size=3, padding=1, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU
            ),
            nn.Conv2d(in_channels//2, 8, 3, padding=1))

        # ConvTranspose2d 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, fpn_fea):
        smooth_fea = self.sc_smooth_layer(fpn_fea)
        shrink = self.shrink_head(smooth_fea)
        centroid = self.centroid_head(smooth_fea)

        param = self.param_head(fpn_fea)

        shift = self.shift_head(fpn_fea)

        if self.training:
            return shrink, centroid, param, shift

        else:
            cate = shrink[0, 0]
            cate_sigmoid = cate.sigmoid()
            h, w = cate_sigmoid.shape

            centroid = centroid[0, 0]
            centroid_sigmoid = centroid.sigmoid()
            final_score = centroid_sigmoid * cate_sigmoid

            # cate_binary = (cate_sigmoid > self.test_cfg.score_thr)
            cate_binary = (cate > 0)

            # img_name = img_meta['img_path'][0].split("/")[-1]
            # cv2.imwrite('shrink_mask/{}'.format(img_name), cate_binary.cpu().numpy() * 255)
            # cv2.imwrite('centroid/{}'.format(img_name), (centroid_sigmoid * cate_binary).cpu().numpy() * 255)

            position = torch.zeros_like(cate_binary)
            connected_area = cc_torch.connected_components_labeling(cate_binary.byte())
            for u in torch.unique(connected_area)[1:]:
                binary = (connected_area == u)
                if torch.sum(binary) < self.test_cfg.min_area:
                    continue
                # if torch.mean(cate_sigmoid[binary]) < self.test_cfg.min_score:
                #     continue

                position += (final_score == torch.max(final_score[binary]))

            position = position.reshape(-1)

            dxy = shift[0].permute(1, 2, 0).reshape(-1, 8)[position]
            param_position = param[0].permute(1, 2, 0).reshape(-1, 2)[position]
            position = torch.where(position)
            Y = position[0] // w
            X = position[0] % w
            dxy[:, ::2] += X[:, None]
            dxy[:, 1::2] += Y[:, None]

            a_up, a_down = param_position[:, 0], param_position[:, 1]

            up_dist = torch.sqrt(torch.pow(dxy[:, 0] - dxy[:, 2], 2) + torch.pow(dxy[:, 1] - dxy[:, 3], 2))[:, None]
            down_dist = torch.sqrt(torch.pow(dxy[:, 4] - dxy[:, 6], 2) + torch.pow(dxy[:, 5] - dxy[:, 7], 2))[:, None]

            sampled_pts_x = torch.linspace(-0.5, 0.5, 11).cuda()

            up_y = (torch.pow(sampled_pts_x, 2)[None, :] * a_up[:, None]) * up_dist
            up_x = (torch.repeat_interleave(sampled_pts_x[None, :], up_y.shape[0], dim=0)) * up_dist
            down_y = (torch.pow(sampled_pts_x, 3)[None, :] * a_down[:, None]) * down_dist
            down_x = (torch.repeat_interleave(sampled_pts_x[None, :], up_y.shape[0], dim=0)) * down_dist

            # 下面进行坐标变换
            up_y = up_y - up_y[:, 0][:, None]
            up_x = up_x - up_x[:, 0][:, None]
            down_y = down_y - down_y[:, 0][:, None]
            down_x = down_x - down_x[:, 0][:, None]

            # 下面将上面的点进行旋转，从而进行缩放
            angle_up = torch.arctan(up_y[:, -1] / up_x[:, -1])
            angle_down = torch.arctan(down_y[:, -1] / down_x[:, -1])
            new_up_x = torch.cos(angle_up)[:, None] * up_x + torch.sin(angle_up)[:, None] * up_y
            new_up_y = torch.cos(angle_up)[:, None] * up_y - torch.sin(angle_up)[:, None] * up_x
            up_y = new_up_y
            up_x = ((up_dist[:, 0] / new_up_x[:, -1])[:, None] * new_up_x)

            new_down_x = torch.cos(angle_down)[:, None] * down_x + torch.sin(angle_down)[:, None] * down_y
            new_down_y = torch.cos(angle_down)[:, None] * down_y - torch.sin(angle_down)[:, None] * down_x
            down_y = new_down_y
            down_x = ((down_dist[:, 0] / new_down_x[:, -1])[:, None] * new_down_x)

            # 下面计算各个曲线需要旋转的角度
            dx_up = dxy[:, 2] - dxy[:, 0]
            dy_up = dxy[:, 3] - dxy[:, 1]
            angle_up = torch.zeros_like(dx_up)
            angle_up[torch.where(dx_up == 0) and torch.where(dy_up > 0)] = np.pi / 2
            angle_up[torch.where(dx_up == 0) and torch.where(dy_up < 0)] = -np.pi / 2
            angle_up[torch.where(dx_up > 0)] = torch.arctan(
                dy_up[torch.where(dx_up > 0)] / dx_up[torch.where(dx_up > 0)])
            angle_up[torch.where(dx_up < 0)] = torch.arctan(
                dy_up[torch.where(dx_up < 0)] / dx_up[torch.where(dx_up < 0)]) + np.pi

            dx_down = dxy[:, 6] - dxy[:, 4]
            dy_down = dxy[:, 7] - dxy[:, 5]
            angle_down = torch.zeros_like(dx_down)
            angle_down[torch.where(dx_down == 0) and torch.where(dy_down > 0)] = np.pi / 2
            angle_down[torch.where(dx_down == 0) and torch.where(dy_down < 0)] = -np.pi / 2
            angle_down[torch.where(dx_down > 0)] = torch.arctan(
                dy_down[torch.where(dx_down > 0)] / dx_down[torch.where(dx_down > 0)])
            angle_down[torch.where(dx_down < 0)] = torch.arctan(
                dy_down[torch.where(dx_down < 0)] / dx_down[torch.where(dx_down < 0)]) + np.pi

            # 下面将各个曲线的11个点旋转到对应位置
            angle_up *= -1
            angle_down *= -1
            new_up_x = torch.cos(angle_up)[:, None] * up_x + torch.sin(angle_up)[:, None] * up_y
            new_up_y = torch.cos(angle_up)[:, None] * up_y - torch.sin(angle_up)[:, None] * up_x

            new_down_x = torch.cos(angle_down)[:, None] * down_x + torch.sin(angle_down)[:, None] * down_y
            new_down_y = torch.cos(angle_down)[:, None] * down_y - torch.sin(angle_down)[:, None] * down_x

            # 下面进行拼接
            final_up_x = dxy[:, 0][:, None] + new_up_x
            final_up_y = dxy[:, 1][:, None] + new_up_y
            final_down_x = dxy[:, 4][:, None] + new_down_x
            final_down_y = dxy[:, 5][:, None] + new_down_y

            final_up_x = final_up_x.cpu()
            final_down_x = final_down_x.cpu()
            final_up_y = final_up_y.cpu()
            final_down_y = final_down_y.cpu()
            final_x_merge = np.concatenate([final_up_x.numpy(), final_down_x.numpy()[:, ::-1]], axis=-1)
            final_y_merge = np.concatenate([final_up_y.numpy(), final_down_y.numpy()[:, ::-1]], axis=-1)
            rebuild_text_masks = np.concatenate([final_x_merge[..., None], final_y_merge[..., None]], axis=-1)

            outputs = dict()
            bboxes = []
            if rebuild_text_masks is not None:
                rebuild_text_masks *= 4
                rebuild_text_masks[:, :, 0] = rebuild_text_masks[:, :, 0] / img_meta['resize_size'][0, 1] * img_meta['ori_size'][0, 1]
                rebuild_text_masks[:, :, 1] = rebuild_text_masks[:, :, 1] / img_meta['resize_size'][0, 0] * img_meta['ori_size'][0, 0]
                bboxes = rebuild_text_masks
            outputs.update({'bboxes': bboxes})

            os.mkdir("masks/{}".format(img_name))
            for i in range(len(bboxes)):
                output_mask = np.zeros(img_meta['ori_size'][0].numpy(), np.uint8)
                cv2.drawContours(output_mask, [bboxes[i].astype(np.int)], 0, 255, -1)
                cv2.imwrite("masks/{}/{}.jpg".format(img_name, i), output_mask)

            return outputs


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def build_decoder(args):
    return Decoder(
        in_channels=args.out_channels
    )