# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math

from utils.misc import (NestedTensor, nested_tensor_from_tensor_list)

from .backbone import build_backbone

from .deformable_transformer import build_deforamble_transformer
import copy


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, hidden_dim, num_feature_levels,
                 train_backbone, train_transformer):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
        """
        super().__init__()
        self.transformer = transformer
        self.num_feature_levels = num_feature_levels
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone

        if not train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        # if self.transformer is not None and not train_transformer:
        if not train_transformer:
            for m in [self.input_proj]:
                for p in m.parameters():
                    p.requires_grad_(False)

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        self.num_channels = hidden_dim


    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
               - outputs.srcs: a list of multi-scale features, the feature of shape [batch_size x H x W x dim]
               - outputs.srcs: a list of multi-scale mask, the mask of shape [batch_size x H x W]

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if self.transformer is not None:
            mask_flatten, memory, pos_embs = self.transformer(srcs, masks, pos, query_embeds)
            shapes = [src.shape for src in srcs]
            srcs = memory.split([h * w for _, _, h, w in shapes], dim=1)
            masks = mask_flatten.split([h * w for _, _, h, w in shapes], dim=1)
        else:
            srcs[0] = F.max_pool2d(srcs[0], kernel_size=4).flatten(2)
            srcs[1] = F.max_pool2d(srcs[1], kernel_size=2).flatten(2)
            srcs = [src.flatten(2).transpose(1, 2) for src in srcs]
            masks = [mask.flatten(1) for mask in masks]
            pos_embs = [p.flatten(2).transpose(1, 2) for p in pos]

        return masks[2], srcs[:-1], pos_embs[2]


def build_detr(args):
    backbone = build_backbone(args)
    train_backbone = args.lr_visu_cnn > 0
    train_transformer = args.lr_visu_tra > 0
    if args.enc_layers > 0:
        transformer = build_deforamble_transformer(args)
    else:
        transformer = None
    model = DeformableDETR(
        backbone,
        transformer,
        hidden_dim=args.hidden_dim,
        num_feature_levels=args.num_feature_levels + 1,
        train_backbone=train_backbone,
        train_transformer=train_transformer
    )
    return model
