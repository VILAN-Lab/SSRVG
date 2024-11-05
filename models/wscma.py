"""Weakly supervised cross modal alignment"""
import torch
from torch import nn
from .otk.layers import OTKernel


# TODO hidden_dim = args.hidden_dim
# TODO out_size = args.max_query_len
# TODO num_feature_levels = args.num_feature_levels

class WSCMA(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_size, num_feature_levels, train_cma):
        super().__init__()
        self.otk_list = nn.ModuleList([OTK_layer(in_dim, hidden_dim, out_size) for _ in range(num_feature_levels)])
        self.proj_list = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_feature_levels)])
        self.img_norm_list = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_feature_levels)])
        self.txt_norm_list = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_feature_levels)])

        if not train_cma:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, img_srcs, txt_src, mask):
        img_feas = []
        txt_feas = []
        for img_src, otk, proj, img_norm, txt_norm in zip(img_srcs, self.otk_list, self.proj_list, self.img_norm_list, self.txt_norm_list):
            img_feas.append(img_norm(proj(img_src)))
            txt_feas.append(txt_norm(otk(txt_src, mask)))

        return img_feas, txt_feas


class OTK_layer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_size):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
        )
        # self.otk = OTKernel(in_dim=hidden_dim, out_size=out_size, heads=1)

    def forward(self, fea, mask):
        # return self.otk(self.proj(fea), mask=~mask)
        return self.proj(fea)


def build_cma(in_dim, args):
    train_cma = args.lr_cma > 0
    return WSCMA(in_dim, args.hidden_dim, args.max_query_len, args.num_feature_levels, train_cma)
