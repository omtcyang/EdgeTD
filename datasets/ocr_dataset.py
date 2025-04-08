# ------------------------------------------------------------------------
# Copyright (2023) Bytedance Ltd. and/or its affiliates
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
from pathlib import Path

import os
import math
import random
import torch
from torch.utils import data
import torch.utils.data
from torch.utils.data import dataset
import torchvision
from torch.utils.data import ConcatDataset
import numpy as np
import datasets.smoothtext_transforms as T
import cv2
from shapely.geometry import Polygon, Point
import copy
from PIL import Image


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, image_set, ann_file, transforms, dataset_name, max_rec_length, max_segmentation_length):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.image_set = image_set
        self.dataset_name = dataset_name
        self.max_rec_length = max_rec_length

        max_segmentation_keys = list(max_segmentation_length.keys())
        current_keys = list(img_folder.parts[-2:])
        current_key = (current_keys[0].split('_')[0] + '_' + current_keys[1].split('_')[0])
        if (current_keys[0].split('_')[0] + '_' + current_keys[1].split('_')[0]) in max_segmentation_keys:
            self.max_segmentation_length = max_segmentation_length[current_key]
        else:
            self.max_segmentation_length = max_segmentation_length['others']

        self._transforms = transforms
        self.parser = ParserLabel(self.image_set, self.dataset_name, self.max_rec_length, self.max_segmentation_length)
        self.get_label = GetTrainingParamsFromPolys(self.image_set)
        print('img_folder==>',img_folder)
        print('max_segmentation_length==>',self.max_segmentation_length)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}

        img, target = self.parser(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
            masks = self.get_label(img, target)
        return img, masks


class ParserLabel(object):
    def __init__(self, image_set='train', dataset_name='', max_rec_length=25, max_segmentation_length=8):
        self.dataset_name = dataset_name
        self.max_rec_length = max_rec_length
        self.max_seg_length = max_segmentation_length
        self.image_set = image_set

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id
        target['dataset_name'] = self.dataset_name

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area
        target["iscrowd"] = iscrowd

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        recog = [obj['rec'][:self.max_rec_length] for obj in anno]
        recog = torch.tensor(recog, dtype=torch.long).reshape(-1, self.max_rec_length)
        target["rec"]  = recog

        poly_pts, poly_inward_pts = [], []
        for obj in anno:
            poly_pts_cur=obj['segmentation']
            assert len(poly_pts_cur)<=self.max_seg_length
            if len(poly_pts_cur)<self.max_seg_length:
                num_points = self.max_seg_length-len(poly_pts_cur)
                pad_seg = [-1]*(num_points-1)+[len(poly_pts_cur)]
                poly_pts_cur+=pad_seg
            poly_pts.append(poly_pts_cur)

            poly_inward_pts_cur=obj['segmentation_inward']
            assert len(poly_inward_pts_cur)<=100
            if len(poly_inward_pts_cur)<100:
                num_points = 100-len(poly_inward_pts_cur)
                pad_seg = [-1]*(num_points-1)+[len(poly_inward_pts_cur)]
                poly_inward_pts_cur+=pad_seg
            poly_inward_pts.append(poly_inward_pts_cur)

        poly_pts, poly_inward_pts = np.array(poly_pts), np.array(poly_inward_pts)
        poly_pts = torch.tensor(poly_pts, dtype=torch.float32).reshape(-1, poly_pts.shape[1])
        poly_inward_pts = torch.tensor(poly_inward_pts, dtype=torch.float32).reshape(-1, poly_inward_pts.shape[1])
        target['poly_pts'], target['poly_inward_pts'] = poly_pts, poly_inward_pts
        target['poly_pts_'], target['poly_inward_pts_'] = poly_pts, poly_inward_pts

        img_valid_mask = np.ones((h,w), dtype=np.uint8)
        target['img_valid_mask'] = Image.fromarray(img_valid_mask)

        if self.image_set == 'train':
            # 生成中心点标签
            # 通过两种不同类型的中心点，到时候做消融试验
            center_pts = torch.zeros(poly_pts.shape[0], 2)
            for i in range(poly_pts.shape[0]):
                poly_pts_i = poly_pts[i]
                if poly_pts_i[-2]==-1: poly_pts_i = poly_pts_i[:int(poly_pts_i[-1])]

                # # 1. 通过中值计算文本中心点
                # xc, yc = polygon_to_center_with_mid(poly_pts_i)
                # center_pts[i][0] = xc
                # center_pts[i][1] = yc  

                # # 2. 通过polygon计算文本中心点
                xc, yc = polygon_to_center_with_PLG(poly_pts_i)   
                center_pts[i][0] = xc
                center_pts[i][1] = yc   

            target['center_pts'] = center_pts.clone().detach().to(dtype=torch.float32)
            # target['center_pts'] = torch.tensor(center_pts, dtype=torch.float32)
        return image, target


class GetTrainingParamsFromPolys(object):
    def __init__(self, image_set='train'):
        self.image_set = image_set

    def __call__(self, img, target):
        if self.image_set == 'val':
            return target

        img_save = (np.array(img.permute(1,2,0))*255).astype(np.uint8)
        h, w, c = img_save.shape

        # 同时生成 【上下双边】 和 【仅中心线】 的拟合参数
        masks = polygon_to_training_label(target, np.array([w,h]))          
        return masks

def dynamic_point(pts, radius):
    theta = random.uniform(0, 1)*2*math.pi
    x_new = pts[0] + radius*math.cos(theta)
    y_new = pts[1] - radius*math.sin(theta)
    return x_new, y_new

def polygon_to_center_with_mid(poly_pts):
    poly_pts_x, poly_pts_y = poly_pts[0::2], poly_pts[1::2]
    minx, maxx, miny, maxy = min(poly_pts_x), max(poly_pts_x), min(poly_pts_y), max(poly_pts_y)
    poly_pts_x, poly_pts_y = poly_pts_x.reshape(2, -1), poly_pts_y.reshape(2, -1)

    xc, yc = -1, -1
    index = poly_pts_x.shape[1]//2
    if poly_pts_x.shape[1]%2==1:
        xc = (poly_pts_x[0, index]+poly_pts_x[1, index])//2
        yc = (poly_pts_y[0, index]+poly_pts_y[1, index])//2

    else:
        xc = ((poly_pts_x[0, index-1]+poly_pts_x[0, index])/2+(poly_pts_x[1, index-1]+poly_pts_x[1, index])/2)//2
        yc = ((poly_pts_y[0, index-1]+poly_pts_y[0, index])/2+(poly_pts_y[1, index-1]+poly_pts_y[1, index])/2)//2

    assert xc>torch.tensor(0, dtype=torch.float32) and yc>torch.tensor(0, dtype=torch.float32)
    return xc, yc


def polygon_to_center_with_PLG(poly_pts):
    poly = Polygon(poly_pts.reshape(-1, 2))
    xc, yc = poly.centroid.xy
    xc, yc = xc[0], yc[0]
    assert xc>torch.tensor(0, dtype=torch.float32) and yc>torch.tensor(0, dtype=torch.float32)
    return int(xc), int(yc)


def polygon_to_training_label(target, scale):
    # poly : [n, 10, 2]       scale:[w,h]
    stride = 4 # 因为是在1/4尺寸的特征图上进行预测
    scale_4s = np.ceil(scale/stride).astype(int)

    polys, polys_inward = target['poly_pts']/stride, target['poly_inward_pts']/stride
    boxes, centers, recs = target['boxes']/stride, target['center_pts']/stride, target['rec']
    polys = (polys.reshape(polys.shape[0], -1, 2)*scale.reshape(-1,2)).reshape(polys.shape[0], -1).numpy().astype(np.int32)

    polys_inward = polys_inward.numpy().astype(np.int32)
    boxes = (boxes.reshape(boxes.shape[0], -1, 2)*scale.reshape(-1,2)).reshape(boxes.shape[0], -1).numpy().astype(np.int32)
    centers = (centers*scale.reshape(-1,2)).numpy().astype(np.int32)

    polys_, polys_inward_ = target['poly_pts_'],target['poly_inward_pts_']

    # masks: ignore/ shrink/ coe_up/ up_head_tail_shift_xy*4/ coe_down/ down_head_tail_shift_xy*4/ centroid/ img_valid_mask
    masks = np.zeros((13, scale_4s[1], scale_4s[0]))
    for p, p_, pi, pi_, center, rec, box in zip(polys, polys_, polys_inward, polys_inward_, centers, recs, boxes):
        channel_masks = 0

        # print('p 1',p, p_)
        if p_[-2]==-1: p = p[:int(p_[-1])].reshape(-1,2)
        else:p = p.reshape(-1,2)
        if pi_[-2]==-1: pi = pi[:int(pi_[-1])].reshape(-1,2)
        else:pi = pi.reshape(-1,2)

        # print('p 2',p, p_)
        binary = np.zeros((masks.shape[1], masks.shape[2]), dtype=np.uint8)
        # cv2.polylines(binary, pi, isClosed=True, color=1, thickness=-1)
        cv2.drawContours(binary, [pi], 0, 1, -1)

        # ========> 1. ignore 
        if rec[0] == 3 and rec[1] == 3 and rec[2] == 3:
            masks[channel_masks] += binary; continue # '###'
        channel_masks+=1

        # ========> 2. shrink mask
        masks[channel_masks] += binary
        channel_masks+=1

        # ========> 3. shift*4 + coe / shift*4 + coe
        pu, pd = p[:int(p.shape[0]//2)], p[int(p.shape[0]//2):]
        pd = pd[::-1]   # 注意，torch.tensor不能做 【::-1】 这个反向排列操作，会报错，需提前转化为np.array
        # # 如果单边轮廓点个数是偶数，加一个点凑成奇数
        # pts_num = pu.shape[0]
        # index = pts_num//2
        # if pts_num%2==0:
        #     if pts_num==2:
        #         pu = np.concatenate([pu[0].reshape(-1,2),((pu[0]+pu[1])/2).reshape(-1,2), pu[1].reshape(-1,2)], axis=0)
        #         pd = np.concatenate([pd[0].reshape(-1,2),((pd[0]+pd[1])/2).reshape(-1,2), pd[1].reshape(-1,2)], axis=0)
        #     else:
        #         pu = np.concatenate([pu[:index].reshape(-1,2),((pu[index-1]+pu[index])/2).reshape(-1,2), pu[index:].reshape(-1,2)], axis=0)
        #         pd = np.concatenate([pd[:index].reshape(-1,2),((pd[index-1]+pd[index])/2).reshape(-1,2), pd[index:].reshape(-1,2)], axis=0)
        # assert pu.shape[0]%2==1 and pd.shape[0]%2==1
        # 点加密
        # pu, pd = linear_interpolation(pu, 5), linear_interpolation(pd, 5)
        x_linspace, y_linspace = np.linspace(0, masks.shape[2]-1, masks.shape[2]), np.linspace(0, masks.shape[1]-1, masks.shape[1])
        x_meshgrid, y_meshgrid = np.meshgrid(x_linspace, y_linspace)
        for num, pud in enumerate([pu,pd]):
            head, tail = pud[0], pud[-1]

            # head tail --> shift*4
            dx_head = (head[0] - x_meshgrid)
            dy_head = (head[1] - y_meshgrid)

            masks[channel_masks] += (binary*dx_head)
            channel_masks+=1
            masks[channel_masks] += (binary*dy_head)
            channel_masks+=1

            dx_tail = (tail[0] - x_meshgrid)
            dy_tail = (tail[1] - y_meshgrid)
            masks[channel_masks] += (binary*dx_tail)
            channel_masks+=1
            masks[channel_masks] += (binary*dy_tail)
            channel_masks+=1

            xs, ys = pud[:,0], pud[:,1]
            # 把中心点序旋转至水平

            angle, dist = angle_dist([xs[0], ys[0]], [xs[-1], ys[-1]])
            xs, ys = Srotate(angle, xs, ys, xs[0], ys[0])
            a_, b_, c_ = np.polyfit(xs, ys, 2)

            # 首先通过拟合获得中点位置来进行下一次线性拟合
            center_x = (xs[0] + xs[-1]) / 2
            center_y = a_ * center_x * center_x + b_ * center_x + c_
            xs = (xs - center_x) / dist
            ys = (ys - center_y) / dist
            a, b, c = np.polyfit(xs, ys, 2)

            masks[channel_masks] += (binary*a)
            channel_masks+=1

        dis_out = np.sqrt(np.power(pi[:, 0][:, None, None] - x_meshgrid[None, :, :], 2)
                          + np.power(pi[:, 1][:, None, None] - y_meshgrid[None, :, :], 2))
        dis_center = np.sqrt(np.power(x_meshgrid - center[0], 2) + np.power(y_meshgrid - center[1], 2))
        masks[channel_masks] += ((np.min(dis_out, axis=0)/(np.min(dis_out, axis=0)+dis_center+1e-6))/2+0.5)*binary

    # masks: ignore/ shrink/ coe_up/ up_head_tail_shift_xy*4/ coe_down/ down_head_tail_shift_xy*4/ centroid/ img_valid_mask
    # for num in range(masks.shape[-1]):
    #     if num==6: print(np.unique(masks[:,:,num]))
    #     if num==11: print(np.unique(masks[:,:,num]))
    #     cv2.imwrite('contours'+str(num)+'.jpg', masks[:,:,num])
    # a=g
    img_valid_mask = np.array(target['img_valid_mask'].resize((masks.shape[2],masks.shape[1])))
    masks = np.concatenate([masks, img_valid_mask.reshape(1,masks.shape[1],masks.shape[2])], axis=0)
    masks = torch.tensor(masks, dtype=torch.float32)

    return masks


def linear_interpolation(pts, num_points):
    """
    线性插值函数，在两个点之间生成均匀分布的新点
    p1: 起始点，形如 (x1, y1)
    p2: 结束点，形如 (x2, y2)
    num_points: 生成的新点数量
    """
    new_points_all = []
    for num_ins in range(pts.shape[0]):
        new_points_cur = []
        for pt_index in range(pts.shape[1]-1):
            p1 = pts[num_ins, pt_index]
            p2 = pts[num_ins, pt_index+1]

            x1, y1 = p1
            x2, y2 = p2

            # 生成均匀分布的参数 t
            t = np.linspace(0, 1, num_points)

            # 对每个参数 t，计算对应的新点坐标
            npts = [[x1 + (x2 - x1) * ti, y1 + (y2 - y1) * ti] for ti in t]
            new_points_cur.append(np.array(npts[:num_points-1]).reshape(-1,2))
        
        # new_points_cur.append(pts[num_ins, -1])
        new_points_cur = np.concatenate(new_points_cur, axis=0)
        new_points_cur = np.concatenate([new_points_cur,np.array(pts[num_ins, -1]).reshape(-1,2)], axis=0)

        new_points_all.append(new_points_cur)

    new_points_all = np.array(new_points_all)

    return new_points_all

def angle_dist(point1, point2):
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    dist = np.sqrt(dy ** 2 + dx ** 2)
    if dx == 0:
        if dy > 0:
            angle = np.pi / 2
        else:
            angle = -np.pi / 2
    elif dx > 0:
        angle = np.arctan(dy / dx)
    else:
        angle = np.arctan(dy / dx) + np.pi
    return angle, dist

def Srotate(angle, valuex, valuey, pointx, pointy):
    valuex = np.array(valuex)
    valuey = np.array(valuey)
    sRotatex = (valuex - pointx) * np.cos(angle) + (valuey - pointy) * np.sin(angle) + pointx
    sRotatey = (valuey - pointy) * np.cos(angle) - (valuex - pointx) * np.sin(angle) + pointy
    return sRotatex, sRotatey

def make_coco_transforms(image_set, max_size_train, min_size_train, max_size_test, min_size_test,
                         crop_min_ratio, crop_max_ratio, crop_prob, rotate_max_angle, rotate_prob,
                         brightness, contrast, saturation, hue, distortion_prob):
    transforms = []
    if image_set == 'train':
        transforms.append(T.RandomSizeCrop(crop_min_ratio, crop_max_ratio, True, crop_prob))
        transforms.append(T.RandomRotate(rotate_max_angle, rotate_prob))
        transforms.append(T.RandomResize(min_size_train, max_size_train))
        transforms.append(T.RandomDistortion(brightness, contrast, saturation, hue, distortion_prob))
    if image_set == 'val':
        transforms.append(T.RandomResize([min_size_test], max_size_test))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(None, None))

    return T.Compose(transforms)

def angle_between_points(A, B, C):
    # Calculate vectors
    AB = (B[0] - A[0], B[1] - A[1])
    BC = (C[0] - B[0], C[1] - B[1])

    # Calculate dot product
    dot_product = AB[0] * BC[0] + AB[1] * BC[1]

    # Calculate magnitudes
    magnitude_AB = math.sqrt(AB[0]**2 + AB[1]**2)
    magnitude_BC = math.sqrt(BC[0]**2 + BC[1]**2)

    # Calculate angle in radians
    cosine_angle = dot_product / (magnitude_AB * magnitude_BC)
    angle_radians = math.acos(cosine_angle)

    # Convert radians to degrees
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees

def build(image_set, args):
    root = Path(args.data_root)
    if image_set == 'train':
        dataset_names = args.train_dataset.split(':')
    elif image_set == 'val':
        dataset_names = args.val_dataset.split(':')
    
    datasets = []
    for dataset_name in dataset_names:
        if dataset_name == 'totaltext_train':
            img_folder = root / "convert_data/totaltext_convert" / "train_images"
            ann_file = root / "convert_data/totaltext_convert" / "train.json"
        elif dataset_name == 'totaltext_val':
            img_folder = root / "convert_data/totaltext_convert" / "test_images"
            ann_file = root / "convert_data/totaltext_convert" / "test.json"
        elif dataset_name == 'mlt_train':
            img_folder = root / "convert_data/mlt2017_convert" / "train_images"
            ann_file = root / "convert_data/mlt2017_convert" / "train.json"
        elif dataset_name == 'ctw1500_train':
            img_folder = root / "convert_data/ctw1500_convert" / "train_images"
            ann_file = root / "convert_data/ctw1500_convert" / "train.json"
        elif dataset_name == 'ctw1500_val':
            img_folder = root / "convert_data/ctw1500_convert" / "test_images"
            ann_file = root / "convert_data/ctw1500_convert" / "test.json"
        elif dataset_name == 'syntext1_train':
            img_folder = root / "convert_data/syntext1_convert" / "train_images"
            ann_file = root / "convert_data/syntext1_convert" / "train.json"
        elif dataset_name == 'syntext2_train':
            img_folder = root / "convert_data/syntext2_convert" / "train_images"
            ann_file = root / "convert_data/syntext2_convert" / "train.json"
        elif dataset_name == 'cocotextv2_train':
            img_folder = root / "convert_data/cocotextv2_convert" / "train_images"
            ann_file = root / "convert_data/cocotextv2_convert" / "train.json"
        elif dataset_name == 'ic13_train':
            img_folder = root / "convert_data/icdar2013_convert" / "train_images"
            ann_file = root / "convert_data/icdar2013_convert" / "train.json"
        elif dataset_name == 'ic15_train':
            img_folder = root / "convert_data/icdar2015_convert" / "train_images"
            ann_file = root / "convert_data/icdar2015_convert" / "train.json"
        elif dataset_name == 'ic13_val':
            img_folder = root / "convert_data/icdar2013_convert" / "test_images"
            ann_file = root / "convert_data/icdar2013_convert" / "test.json"
        elif dataset_name == 'ic15_val':
            img_folder = root / "convert_data/icdar2015_convert" / "test_images"
            ann_file = root / "convert_data/icdar2015_convert" / "test.json"
        elif dataset_name == 'inversetext':
            img_folder = root / "convert_data/inversetext_convert" / "test_images"
            ann_file = root / "convert_data/inversetext_convert" / "test.json"
        else:
            raise NotImplementedError

        print('args.max_size_train, args.min_size_train',args.max_size_train, args.min_size_train)
        transforms = make_coco_transforms(image_set, args.max_size_train, args.min_size_train,
              args.max_size_test, args.min_size_test, args.crop_min_ratio, args.crop_max_ratio,
              args.crop_prob, args.rotate_max_angle, args.rotate_prob, args.brightness, args.contrast,
              args.saturation, args.hue, args.distortion_prob)
        dataset = CocoDetection(img_folder, image_set, ann_file, 
                                transforms=transforms, dataset_name=dataset_name, max_rec_length=args.max_rec_length,
                                max_segmentation_length=args.max_segmentation_length)
        datasets.append(dataset)

    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)

    return dataset
