import copy
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .visual_model.clip import build_clip
from .visual_model.backbone import build_backbone
from .language_model.bert import build_bert
from .language_model.position_encoding import build_position_encoding
from .score_module import build_score_module
from .cross_attention import build_mod, build_ca_layer, build_attn_pool
from .visual_model.sampling import sampling, sampling_swin, sampling_for_three
from utils.misc import NestedTensor
# from .visual_model.position_encoding import build_position_encoding

class ZMod(nn.Module):
    def __init__(self, args):
        super(ZMod, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.multi_scale = args.multi_scale
        self.num_text_token = args.max_query_len
        #  clip
        self.clip = build_clip('RN50x64')

        self.backbone = build_backbone(args)
        self.bert = build_bert(args)
        self.cimg_proj = nn.Conv1d(4096, self.hidden_dim, kernel_size=1)

        # sig_lip
        # way = "/home/user/D/32T/zsy/tmm_msvlt/checkpoints/siglip-so400m-patch14-384"
        # self.siglip = SiglipVisionModel.from_pretrained(way)
        # self.siglip_processor = AutoProcessor.from_pretrained(way)
        # for p in self.siglip.parameters():
        #     p.requires_grad = False
        # self.sigimg_proj = nn.Conv1d(1152, self.hidden_dim, kernel_size=1)
        # self.sig_position_embedding = build_position_encoding(self.hidden_dim, 729)

        self.txt_proj = nn.Linear(self.bert.num_channels, self.hidden_dim)
        self.position_embedding = build_position_encoding(self.hidden_dim, self.num_text_token)
        self.hmod = build_mod(args)
        self.cmod = build_mod(args)
        self.bbox_embed_first = MLP(self.hidden_dim, self.hidden_dim, 4, 3)
        self.bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 4, 3)
        self.erase = random_erase(args.erase)
        self.head_token = nn.Embedding(1, self.hidden_dim)
        self.context_token = nn.Embedding(1, self.hidden_dim)
        self.wpa_loss = args.wpa_loss
        self.aux_loss = args.aux_loss
        self.backbone_name = args.backbone

        self.keep_dim = True

        if self.multi_scale:
            self._multiscale_init(args)
        else:
            self._singlescale_init(args)

    def _singlesckale_init(self, args):
        divisor = [32]
        self.num_img_token = [int((args.imsize / d) ** 2) for d in divisor]

    def _multiscale_init(self, args):
        divisor = [16, 32, 64]
        self.num_img_token = [int((args.imsize / d) ** 2) for d in divisor]
        self.img_scale = [int(args.imsize / d) for d in divisor]

        if "swin" in args.backbone or "resnet101" in args.backbone:
            number = 5 if self.keep_dim else 3
            if self.keep_dim:
                self.sampling_module = sampling_swin(args.hidden_dim)
            else:
                self.sampling_module = sampling_for_three(args.hidden_dim)
            self.v_proj = nn.Conv1d(self.hidden_dim * number, self.hidden_dim, kernel_size=1)
            self.vh_proj = nn.Linear(self.hidden_dim * number, self.hidden_dim)
            self.vc_proj = nn.Linear(self.hidden_dim * number, self.hidden_dim)
            self.score_module = nn.ModuleList([build_score_module(args) for i in range(number)])
            self.encoder_ca = nn.ModuleList([build_ca_layer(args) for _ in range(number)])
            # self.ca = nn.ModuleList([build_ca_layer(args) for _ in range(5)])
        else:
            self.sampling_module = sampling(args.hidden_dim)
            self.v_proj = nn.Conv1d(self.hidden_dim * 3, self.hidden_dim, kernel_size=1)
            self.vh_proj = nn.Linear(self.hidden_dim * 3, self.hidden_dim)
            self.vc_proj = nn.Linear(self.hidden_dim * 3, self.hidden_dim)
            self.score_module = nn.ModuleList([build_score_module(args) for _ in range(3)])
            self.ca = nn.ModuleList([build_ca_layer(args) for _ in range(3)])

        self.th_pool = build_attn_pool(args)
        self.tc_pool = build_attn_pool(args)

    def forward(self, img_data, text_data, r1_mask):
        keep_dim = self.keep_dim
        bs = img_data.tensors.shape[0]
        # visual backbone
        img_srcs, img_pos = self.backbone(img_data)

        if "swin" in self.backbone_name or "resnet101" in self.backbone_name:
            if keep_dim:
                img_pos = img_pos[-2]
                img_mask = img_srcs[-2].mask
                img_srcs = [src.tensors for src in img_srcs]
            else:
                img_poss = img_pos[-3:]
                img_masks = [i.mask for i in img_srcs[-3:]]
                img_srcs = [src.tensors for src in img_srcs[-3:]]
        else:
            img_pos = img_pos[-1]
            img_mask = img_srcs[-1].mask
            img_srcs = [src.tensors for src in img_srcs[1:]]


        if keep_dim:
            img_mask, img_pos = img_mask.flatten(-2), img_pos.flatten(-2).transpose(1, 2)
            img_srcs = self.sampling_module(img_srcs)
        else:
            img_masks, img_poss = [i.flatten(-2) for i in img_masks], [i.flatten(-2).transpose(1, 2) for i in img_poss]
            v_imb_backup = self.sampling_module(img_srcs)
            img_srcs = [src.flatten(-2) for src in img_srcs]

        # img_srcs = self.sampling_module(img_srcs)

        # siglip_src = self.siglip(pixel_values=sig_img).last_hidden_state.permute([0, 2, 1])
        # siglip_src = self.sigimg_proj(siglip_src)
        # siglip_pos = self.sig_position_embedding(siglip_src)
        clip_src = self.clip(img_data.tensors).flatten(2)
        clip_src = self.cimg_proj(clip_src)
        # cost = cost_matrix_cosine(img_srcs[-1].transpose(1, 2), clip_src.transpose(1, 2))
        # cost = (cost * ~img_mask).sum() / (~img_mask).sum()
        # language bert
        txt_fea = self.bert(text_data)
        txt_src, txt_mask = txt_fea.decompose()
        txt_src = self.txt_proj(txt_src)
        txt_pos = self.position_embedding(txt_src)
        assert txt_mask is not None

        if self.multi_scale:
            # split head word and context
            ctxt_mask = txt_mask | ~r1_mask
            htxt_mask = r1_mask
            htxt_mask[:, 0] = False

            if keep_dim:
                txt_srcs = [module(clip_src, img_pos, img_mask, txt_src, txt_mask)
                # txt_srcs = [module(siglip_src, siglip_pos, img_mask, txt_src, txt_mask)
                           for module in self.score_module]
            else:
                txt_srcs = [module(clip_src, img_poss[-2], img_masks[-2], txt_src, txt_mask)
                           for module in self.score_module]



            # htxt_srcs = [src for src in txt_srcs[: -1]]
            # ctxt_srcs = [src for src in txt_srcs[1:]]
            # cross attn

            # 考虑把尺度信息保留 特别是大尺度的
            if keep_dim:
            # img_srcs:[list5 (batch,256,20*20)]  txt_src batch,15,256    vt_embs:8,400,256
                vt_embs = [module(img_src, img_pos, img_mask, txt_src, txt_pos, txt_mask)
                           for module, img_src, txt_src
                           # in zip(self.ca, img_srcs, txt_srcs)]
                           in zip(self.encoder_ca, img_srcs, txt_srcs)]
            #
            else:
                vt_embs = [module(img_src, img_pos, img_mask, txt_src, txt_pos, txt_mask)
                           for module, img_src, img_pos, img_mask, txt_src
                           in zip(self.encoder_ca, img_srcs, img_poss, img_masks, txt_srcs)]
                vt_embs[0] = self.linear1(vt_embs[0].permute([0, 2, 1])).permute([0, 2, 1])
                vt_embs[-1] = self.linear2(vt_embs[-1].permute([0, 2, 1])).permute([0, 2, 1])

            # decoder
            txt_emb = [src[1] for src in txt_srcs]
            if self.keep_dim:
                v_emb = self.v_proj(torch.cat(img_srcs, dim=1)).transpose(1, 2)  # 图像特征
            else:
                v_emb = self.v_proj(torch.cat(v_imb_backup, dim=1)).transpose(1, 2)
            vh_emb = self.vh_proj(torch.cat(vt_embs, dim=-1))  # 多模态特征
            # vc_emb = self.vc_proj(torch.cat(vt_embs, dim=-1))
            th_emb = self.th_pool(torch.cat(txt_emb, dim=-1), txt_mask, htxt_mask)
            tc_emb = self.tc_pool(torch.cat(txt_emb, dim=-1), txt_mask, ctxt_mask)
            # h_tgt, c_tgt = self.mod(self.object_token.weight.unsqueeze(0).repeat(bs, 1, 1),
            #                         th_emb, tc_emb, vh_emb, vc_emb, v_emb,
            #                         htxt_mask, ctxt_mask, img_masks[1], txt_pos, img_pos[1])

            if not self.keep_dim:
                img_mask = img_masks[-2]
                img_pos = img_poss[-2]
            h_tgt, h_map = self.hmod(self.head_token.weight.unsqueeze(0).repeat(bs, 1, 1),
                              tc_emb, vh_emb, v_emb, img_mask, img_pos)
            a = h_tgt[-1:, 0, 0, :].unsqueeze(0)
            c_tgt, c_map = self.cmod(a, th_emb, vh_emb, v_emb, img_mask, img_pos)


            # h_tgt, h_map = self.hmod(self.head_token.weight.unsqueeze(0).repeat(bs, 1, 1),
            #                   th_emb, vh_emb, v_emb, img_mask, img_pos)
            # c_tgt, c_map = self.cmod(self.context_token.weight.unsqueeze(0).repeat(bs, 1, 1),
            #                   tc_emb, vc_emb, v_emb, img_mask, img_pos)

            tgt_first = h_tgt.squeeze(2)
            tgt = c_tgt.squeeze(2)

            # tgt = self.erase(h_tgt.squeeze(2), c_tgt.squeeze(2))

            coord_first = self.bbox_embed_first(tgt_first).sigmoid()
            bbox_first = {'pred_boxes': coord_first[-1]}
            coord = self.bbox_embed(tgt).sigmoid()
            bbox = {'pred_boxes': coord[-1]}
            if self.aux_loss:
                bbox['aux_outputs'] = self._set_aux_loss(coord)
                bbox['aux_outputs_first'] = self._set_aux_loss(coord_first)

            output = {'bbox': bbox,
                      'bbox_first': bbox_first,
                      'inter_s': None,
                      'pos_dist': 0,
                      'neg_dist': 0,
                      'h_map': h_map,
                      'c_map': c_map,
                      }
            return output

    @torch.jit.unused
    def _set_aux_loss(self, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_boxes': b}
                for b in zip(outputs_coord[:-1])]


# def sampling(input, operator='updown'):
#     temp = []
#     if input[0].size(2) != input[-1].size(2):
#         if operator is 'up':
#             src = input[-1]
#             bs, c, n = src.shape
#             src = src.unsqueeze(2).view(bs, c, int(n ** 0.5), -1)
#             temp.append(input[0])
#             temp.append(F.interpolate(src, scale_factor=2, mode='bilinear').flatten(2))
#         elif operator is 'down':
#             src = input[0]
#             bs, c, n = src.shape
#             src = src.unsqueeze(2).view(bs, c, int(n ** 0.5), -1)
#             temp.append(F.max_pool2d(src, kernel_size=2).flatten(2))
#             temp.append(input[-1])
#         else:
#             src = input[0]
#             bs, c, n = src.shape
#             src = src.unsqueeze(2).view(bs, c, int(n ** 0.5), -1)
#             temp.append(F.max_pool2d(src, kernel_size=2).flatten(2))
#             temp.append(input[1])
#             src = input[-1]
#             bs, c, n = src.shape
#             src = src.unsqueeze(2).view(bs, c, int(n ** 0.5), -1)
#             temp.append(F.interpolate(src, scale_factor=2, mode='bilinear').flatten(2))
#         return temp
#     else:
#         if operator is 'up':
#             src = input[0]
#             bs, c, n = src.shape
#             src = src.unsqueeze(2).view(bs, c, int(n ** 0.5), -1)
#             temp.append(F.interpolate(src, scale_factor=2, mode='bilinear').flatten(2))
#             temp.append(input[-1])
#         elif operator is 'down':
#             src = input[-1]
#             bs, c, n = src.shape
#             src = src.unsqueeze(2).view(bs, c, int(n ** 0.5), -1)
#             temp.append(input[0])
#             temp.append(F.max_pool2d(src, kernel_size=2).flatten(2))
#         return temp


class WeightedActivations(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize learnable weights a, b, and c
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))
        self.c = nn.Parameter(torch.randn(1))

    def forward(self, x):
        # Apply the weighted combination of activation functions
        return self.a * F.relu(x) + self.b * F.leaky_relu(x) + self.c * F.tanh(x)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        # self.activate = WeightedActivations()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            # x = self.activate(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class random_erase(nn.Module):
    def __init__(self, drop_prob):
        super(random_erase, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x, y):
        assert x.dim() == y.dim()
        assert x.shape == y.shape
        if self.drop_prob < 0.0 or self.drop_prob > 1.0:
            raise ValueError(f"drop probability has to be between 0 and 1, but got {self.drop_prob}")
        if not self.training or self.drop_prob == 0.0:
            return (x + y) / 2
        layer, bs, _ = x.shape

        survival_rate = 1.0 - self.drop_prob
        size = [layer, bs, 1]
        noise_pos = torch.empty(size, dtype=x.dtype, device=x.device)
        noise_src = torch.empty(size, dtype=x.dtype, device=x.device)
        noise_pos = noise_pos.bernoulli_(survival_rate).to(torch.bool)
        noise_src = noise_src.bernoulli_(0.5).to(torch.bool)
        noise_x = noise_pos | noise_src
        noise_y = ~(noise_pos ^ noise_x)
        output = (x * noise_x + y * noise_y) / (noise_x.to(torch.int) + noise_y.to(torch.int))
        return output


def cost_matrix_cosine(x, y, eps=1e-8):
    """Compute cosine distnace across every pairs of x, y (batched)
    [B, L_x, D] [B, L_y, D] -> [B, Lx, Ly]"""
    assert x.dim() == y.dim()
    assert x.size(0) == y.size(0)
    assert x.size(2) == y.size(2)
    # x = torch.exp(x)
    # y = torch.exp(y)
    x_norm = F.normalize(x, p=2, dim=-1, eps=eps)
    y_norm = F.normalize(y, p=2, dim=-1, eps=eps)

    cosine_sim = x_norm.matmul(y_norm.transpose(1, 2))
    # cosine_sim = x.matmul(y.transpose(1, 2))
    cosine_dist = torch.diagonal(cosine_sim, dim1=1, dim2=2)
    return cosine_dist

