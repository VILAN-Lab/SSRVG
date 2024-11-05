# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model.
"""
import torch
import torch.nn.functional as F
from torch import nn

from .cross_attention import build_mod


class DETR_decoder(nn.Module):
    """ This is the DETR module that performs object detection """

    def __init__(self, transformer, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.aux_loss = aux_loss

    def forward(self, src, mask, pos_embed, tgt, tgt_pos_embed, tgt_mask):
        hs = self.transformer(src, mask, pos_embed, tgt_pos_embed, tgt, tgt_mask)[0]

        outputs_coord = self.bbox_embed(hs[:, :, :1]).sigmoid().squeeze(2)
        out = {'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_boxes': b}
                for b in zip(outputs_coord[:-1])]


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


def build_decoder(args):
    transformer = build_mod(args)
    model = DETR_decoder(transformer, num_queries=args.num_subj_token, aux_loss=args.aux_loss)
    return model
