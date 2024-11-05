# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from utils.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, num_channels: int, return_interm_layers: bool, model_name):
        super().__init__()
        if model_name == "swin-base":
            self.body = backbone
        elif model_name == "swin-tiny":
            self.body = backbone
        elif model_name == "swin-small":
            self.body = backbone
        elif model_name == "resnet101":
            self.body = backbone
        else:
            for name, parameter in backbone.named_parameters():
                if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                    parameter.requires_grad_(False)
            if return_interm_layers:
                return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            else:
                return_layers = {'layer4': "0"}
            self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

from models.visual_model.swin_transformer import SwinTransformer
from models.visual_model.fpn import FPN

class swin_neck(nn.Module):
    def __init__(self, name):
        super(swin_neck, self).__init__()
        if "base" in name:
            self.neck = FPN(in_channels=[128, 256, 512, 1024],out_channels=256,num_outs=5)
            self.backbone = SwinTransformer(
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
            ape=False,
            drop_path_rate=0.3,
            patch_norm=True,
            use_checkpoint=False)
        elif "small" in name:
            self.neck = FPN(in_channels=[96, 192, 384, 768],out_channels=256,num_outs=5)
            self.backbone = SwinTransformer(
            embed_dim=96,
            depths=[2, 2, 18, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            ape=False,
            drop_path_rate=0.2,
            patch_norm=True,
            use_checkpoint=False)
        elif "tiny" in name:
            self.neck = FPN(in_channels=[96, 192, 384, 768],out_channels=256,num_outs=5)
            self.backbone = SwinTransformer(
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            ape=False,
            drop_path_rate=0.2,
            patch_norm=True,
            use_checkpoint=False)
        else:
            pass

        # checkpoint = torch.load("./checkpoints/cascade_mask_rcnn_swin_base_patch4_window7.pth")
        # a,b = self.load_state_dict(checkpoint['state_dict'], strict=False)
        # print(a)

    def forward(self, img):
        x = self.backbone(img)  # (8,128,160,160),(8,256,80,80),(8,512,40,40),(8,1024,20,20)
        x = self.neck(x)
        output = {}
        for i in range(len(x)):
            output["%s" %i] = x[i]

        return output

class resnet_neck(nn.Module):
    def __init__(self, name: str, dilation: bool):
        super(resnet_neck, self).__init__()
        self.neck = FPN(in_channels=[256, 512, 1024, 2048],out_channels=256,num_outs=5)
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=False, norm_layer=FrozenBatchNorm2d)
        return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
        # model_without_ddp.backbone[0].body.backbone  .layer1

    def forward(self, img):
        x = self.backbone(img)  # (8,128,160,160),(8,256,80,80),(8,512,40,40),(8,1024,20,20)
        x = [x[i] for i in x]
        x = self.neck(x)
        output = {}
        for i in range(len(x)):
            output["%s" %i] = x[i]
        return output

class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 return_interm_layers: bool,
                 dilation: bool, train_cnn: bool):
        # TODO pretrained backbone
        if name == "swin-base" or name == "swin-tiny" or name == "swin-small":
            # backbone = SwinForImageClassification.from_pretrained("./checkpoints/swin-base-patch4-window7-224-in22k-model")
            backbone = swin_neck(name)
        elif name == "resnet101":
            backbone = resnet_neck(name, dilation)
        elif name == 'resnet152':
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=False, norm_layer=FrozenBatchNorm2d)
            backbone.load_state_dict(torch.load('./checkpoints/resnet152_pretrained.pth'))
                # pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        if not train_cnn:
            for p in backbone.parameters():
                p.requires_grad = False
        assert name in ('resnet50', 'resnet101', 'swin-base', 'swin-tiny', 'swin-small', 'resnet152')
        num_channels = 2048
        super().__init__(backbone, num_channels, return_interm_layers, name)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    return_interm_layers = True
    train_cnn = args.lr_visu_cnn > 0
    backbone = Backbone(args.backbone, return_interm_layers, args.dilation, train_cnn)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
