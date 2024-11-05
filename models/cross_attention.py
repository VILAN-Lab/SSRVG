# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .attention.multi_head_attention import MultiheadAttention

"""Modulated Object Detection"""


class MOD(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1, activation="gelu",
                 return_intermediate_dec=True):
        super().__init__()

        decoder_layer = DecoderLayer(d_model, nhead, dim_feedforward,
                                     dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = Decoder(decoder_layer, num_decoder_layers, decoder_norm,
                               return_intermediate=return_intermediate_dec)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead
        self.norm = nn.LayerNorm(d_model)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt, t_emb, vt_emb, v_emb,
                     v_mask: Optional[Tensor] = None,
                     v_pos: Optional[Tensor] = None):
        tgt = tgt.transpose(0, 1)
        t_emb = self.norm(t_emb.unsqueeze(0))
        vt_emb = self.norm(vt_emb.transpose(0, 1))
        v_emb = self.norm(v_emb.transpose(0, 1))
        v_pos = v_pos.transpose(0, 1)
        hs, att_map = self.decoder(tgt, t_emb, vt_emb, v_emb,
                          v_mask, v_pos)
        return hs.transpose(1, 2), att_map

    # def forward(self, object_token, th_emb, tc_emb,
    #             vh_emb, vc_emb, v_emb,
    #             img_mask, img_pos):
    #     h_tgt = self.forward_once(object_token, th_emb, vh_emb,
    #                               v_emb, img_mask, img_pos)
    #     c_tgt = self.forward_once(object_token, tc_emb, vc_emb,
    #                               v_emb, img_mask, img_pos)
    #     return h_tgt, c_tgt


class AttentionPool(nn.Module):
    def __init__(self, query_len, d_model, nhead, pooling_number):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(query_len, d_model) / d_model ** 0.5)
        self.src_proj = nn.Linear(d_model * pooling_number, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.num_heads = nhead

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask, split_mask):
        # NCHW -> (HW)NC
        bs = mask.shape[0]
        x_mean = torch.sum(x[:, 1:] * ~split_mask[:, 1:].unsqueeze(-1), dim=1) / torch.sum(~split_mask[:, 1:],
                                                                                           dim=1).unsqueeze(-1)
        x = torch.cat([x_mean.unsqueeze(1), x[:, 1:]], dim=1)
        x = x.transpose(0, 1)
        x = F.gelu(self.src_proj(x))
        x = x + self.positional_embedding[:, None, :].to(x.dtype)
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            key_padding_mask=mask,
            need_weights=False
        )
        return x[0]


class CA_module(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_ca_layers=2,
                 dim_feedforward=2048, dropout=0.1, activation="gelu",
                 return_intermediate_dec=False):
        super().__init__()
        ca_layer = AttentionLayer(d_model, nhead, dim_feedforward, dropout, activation)
        ca_norm = nn.LayerNorm(d_model)
        self.ca = AttentionModule(ca_layer, num_ca_layers, ca_norm,
                                  return_intermediate=return_intermediate_dec)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead
        self.norm = nn.LayerNorm(d_model)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, img_src, img_pos, img_mask, txt_src, txt_pos, txt_mask):
        txt_emb = txt_src[1]
        score = txt_src[0]
        img_src = self.norm(img_src.permute(2, 0, 1))
        txt_emb = self.norm(txt_emb.transpose(0, 1))
        img_pos = img_pos.transpose(0, 1)
        txt_pos = txt_pos.transpose(0, 1)
        score = score.permute(2, 0, 1)
        emb = self.ca(img_src, txt_emb, txt_emb, score,
                      img_pos, txt_pos, img_mask, txt_mask)[-1].transpose(0, 1)
        return emb


class AttentionModule(nn.Module):

    def __init__(self, attn_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(attn_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self,
                q, k, v, s,
                pos_q: Optional[Tensor] = None,
                pos_k: Optional[Tensor] = None,
                mask_q: Optional[Tensor] = None,
                mask_k: Optional[Tensor] = None):
        output = q

        intermediate = []

        for layer in self.layers:
            output = layer(output, k, v, s,
                           pos_q, pos_k, mask_q, mask_k)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class AttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,
                q, k, v, s,
                pos_q: Optional[Tensor] = None,
                pos_k: Optional[Tensor] = None,
                mask_q: Optional[Tensor] = None,
                mask_k: Optional[Tensor] = None):
        q2 = self.cross_attn(q, k, value=v, score=s,
                             key_padding_mask=mask_k)[0]
        q = q + self.dropout1(q2)
        q = self.norm1(q)
        q2 = self.self_attn(query=self.with_pos_embed(q, pos_q),
                            key=self.with_pos_embed(q, pos_q),
                            value=q, key_padding_mask=mask_q)[0]
        q = q + self.dropout2(q2)
        q = self.norm2(q)
        q2 = self.linear2(self.dropout(self.activation(self.linear1(q))))
        q = q + self.dropout3(q2)
        q = self.norm3(q)
        return q


class Decoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, t_emb, vt_emb, v_emb,
                v_mask: Optional[Tensor] = None,
                v_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []
        intermediate_map = []

        for layer in self.layers:
            output, att_map = layer(tgt, t_emb, vt_emb, v_emb,
                              v_mask, v_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                intermediate_map.append(att_map)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_map)

        return output.unsqueeze(0)


class DecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # self.txt_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.img_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.droppath = DropPath(0.2, 'row')

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, t_emb, vt_emb, v_emb,
                v_mask: Optional[Tensor] = None,
                v_pos: Optional[Tensor] = None):
        # tgt2 = self.txt_attn(tgt,
        #                      key=self.with_pos_embed(t_emb, t_pos),
        #                      value=t_emb,
        #                      key_padding_mask=t_mask)[0]
        # tgt = tgt + self.dropout1(tgt2)
        # tgt = self.norm1(tgt)
        tgt2, att_map = self.img_attn(query=self.with_pos_embed(tgt, t_emb),
                             key=self.with_pos_embed(vt_emb, v_pos),
                             value=self.with_pos_embed(v_emb, v_pos),
                             key_padding_mask=v_mask)
        tgt = tgt + self.droppath(self.dropout2(tgt2))
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.droppath(self.dropout3(tgt2))
        tgt = self.norm3(tgt)
        return tgt, att_map

    def forward_pre(self, tgt, t_emb, vt_emb, v_emb,
                    t_mask: Optional[Tensor] = None,
                    v_mask: Optional[Tensor] = None,
                    t_pos: Optional[Tensor] = None,
                    v_pos: Optional[Tensor] = None):
        tgt2 = self.norm2(tgt)
        tgt2 = self.img_attn(query=tgt2,
                             key=self.with_pos_embed(vt_emb, v_pos),
                             value=self.with_pos_embed(v_emb, v_pos),
                             key_padding_mask=v_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_ca_layer(args):
    model = CA_module(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_ca_layers=args.ca_layers,
        dim_feedforward=args.dim_feedforward,
        activation=args.activation,
    )
    _reset_parameters(model)
    return model


def build_mod(args):
    return MOD(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_decoder_layers=args.dec_layers,
        activation=args.activation,
        return_intermediate_dec=True,
    )


def build_attn_pool(args):
    return AttentionPool(
        query_len=args.max_query_len,
        d_model=args.hidden_dim,
        nhead=args.nheads,
        pooling_number=5 if "swin" in args.backbone or "resnet101" in args.backbone else 3
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def _reset_parameters(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


class DropPath(nn.Module):
    def __init__(self, drop_prob, mode: str):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.mode = mode

    def forward(self, x):
        return stochastic_depth(x, self.drop_prob, self.mode, self.training)


def stochastic_depth(input: Tensor, p: float, mode: str, training: bool = True) -> Tensor:
    if p < 0.0 or p > 1.0:
        raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
    if mode not in ["batch", "row"]:
        raise ValueError(f"mode has to be either 'batch' or 'row', but got {mode}")
    if not training or p == 0.0:
        return input

    survival_rate = 1.0 - p
    if mode == "row":
        size = [input.shape[1]] + [1] * (input.ndim - 1)
    else:
        size = [1] * input.ndim
    noise = torch.empty(size, dtype=input.dtype, device=input.device)
    noise = noise.bernoulli_(survival_rate).transpose(0, 1)
    if survival_rate > 0.0:
        noise.div_(survival_rate)
    return input * noise
