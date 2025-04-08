# ------------------------------------------------------------------------
# Copyright (2023) Bytedance Ltd. and/or its affiliates
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch
import torch.nn.functional as F
from torch import nn

from util.misc_smoothtext import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .neck import build_neck
from .decoder import build_decoder
from .criterion import build_criterion

import pdb

class SmoothText(nn.Module):
    def __init__(self, backbone, neck, decoder):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes

        """
        super().__init__()

        self.backbone = backbone
        self.neck = neck
        self.decoder = decoder

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
               - sequences : box_seq,coe_a_updown_seq,head_tail_seq,label_seq
               - box_seq: the input sequence for locating in the first decoder layer.
               - coe_a_updown_seq: the input sequence for param predicting in the second decoder layer.
               - head_tail_seq: the input sequence for head tail coords predicting in the second decoder layer.
               - label_seq: the input sequence for recognizing in the second decoder layer.

            It returns a dict with the following elements:
               - out: prediction of location and recognition 
        """

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        feas = self.backbone(samples)
        feas_src = {key:feas[key].decompose()[0] for key in list(feas.keys())}
        feas_mask = {key:feas[key].decompose()[1] for key in list(feas.keys())}

        assert feas_mask is not None

        feas_fpn = self.neck(feas_src)
        feas_names, feas_concates = [], []
        for name, x in feas_fpn.items():
            feas_names.append(name)
            feas_concates.append(x)
            
        preds = self.decoder(feas_concates[0])

        if preds == None:
            return None

        if self.training:
            return preds
        else:
            return outputs

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):

    device = torch.device(args.device)

    backbone = build_backbone(args)
    neck = build_neck(args)
    decoder = build_decoder(args)

    model = SmoothText(backbone, neck, decoder)

    criterion = build_criterion(args)
    criterion.to(device)

    return model, criterion
