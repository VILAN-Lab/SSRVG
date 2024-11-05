import torch
from torch import nn
import torch.nn.functional as F


class score_module(nn.Module):

    def __init__(self, d_model, nhead, dropout, alpha, sigma):
        super().__init__()
        # self.img_proj = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.img_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # self.img_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.img_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.txt_proj = nn.Linear(d_model, d_model)
        self.img_norm = nn.LayerNorm(d_model)
        self.alpha = nn.Parameter(torch.Tensor([alpha]))
        self.sigma = nn.Parameter(torch.Tensor([sigma]))

        # self.cov1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # self.cov2 = nn.AvgPool2d(kernel_size=4, stride=4)
        # self.a = nn.Parameter(torch.tensor(1.0))
        # self.b = nn.Parameter(torch.tensor(1.0))

        # self.sc_alpha = nn.Parameter(torch.Tensor([alpha]))
        # self.sc_sigma = nn.Parameter(torch.Tensor([sigma]))
        self.sigmoid = nn.Sigmoid()
        # self.linear = nn.Linear(400, out)

        #two_level

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, img_emb, img_pos, img_mask, txt_emb, txt_mask):
        bs = txt_emb.shape[0]
        # img_emb = self.img_proj(img_emb).permute(2, 0, 1)
        img_emb = img_emb.permute(2, 0, 1)
        arg_emb = torch.sum(img_emb, dim=0) / torch.sum(~img_mask, dim=-1).unsqueeze(-1)
        txt_emb = torch.cat([arg_emb.unsqueeze(1), txt_emb[:, 1:]], dim=1).transpose(0, 1)
        img_pos = img_pos.transpose(0, 1)
        img_emb = img_emb + img_pos
        # img_emb1 = img_emb.permute([2, 1, 0]).view(256, bs, 20, 20)
        # img_emb2 = img_emb.permute([2, 1, 0]).view(256, bs, 20, 20)
        # arg_emb = torch.cat([arg_emb.unsqueeze(0), img_emb], dim=0)
        # img_mask = torch.cat([torch.Tensor([[0]]).repeat(bs, 1).to(torch.bool).to(img_mask.device), img_mask], dim=-1)
        # img_emb1 = self.cov1(img_emb1).view(256, bs, -1).permute([2, 1, 0])
        # img_emb2 = self.cov2(img_emb2).view(256, bs, -1).permute([2, 1, 0])
        img_emb = self.img_attn(query=img_emb, key=txt_emb, value=txt_emb,
                                key_padding_mask=txt_mask)[0]
        # img_emb_1 = self.img_attn1(query=img_emb1, key=txt_emb, value=txt_emb,
        #                         key_padding_mask=txt_mask)[0]
        # img_emb_2 = self.img_attn2(query=img_emb2, key=txt_emb, value=txt_emb,
        #                         key_padding_mask=txt_mask)[0]
        # arg_emb = arg_emb.transpose(0, 1)
        img_emb = img_emb.transpose(0, 1)
        # img_emb_1 = img_emb_1.transpose(0, 1)
        # img_emb_2 = img_emb_2.transpose(0, 1)
        txt_emb = txt_emb.transpose(0, 1)
        txt_emb = self.txt_proj(txt_emb)

        score = cost_matrix_cosine(img_emb, txt_emb)
        # score1 = cost_matrix_cosine(img_emb_1, txt_emb) # 8 100 15
        # score2 = cost_matrix_cosine(img_emb_2, txt_emb) # 8 25 15

        # score1 = score1.view(bs, 10, 10, -1).repeat(1, 2, 2, 1).view(bs, 400, -1)
        # score2 = score2.view(bs, 5, 5, -1).repeat(1, 4, 4, 1).view(bs, 400, -1)
        # score = self.a * score + self.b * score1
        # joint_head_mask = img_mask.unsqueeze(-1) | htxt_mask.unsqueeze(-2)
        # joint_context_mask = img_mask.unsqueeze(-1) | ctxt_mask.unsqueeze(-2)

        score = self.alpha * torch.exp(-0.5 * (1 - score).pow(2) / (self.sigma ** 2))

        # siglipp
        score = score.permute(0, 2, 1)  # 变成 (1, 256, 729)

        # 使用线性插值将 729 维度变为 400
        output = F.interpolate(score, size=400, mode='linear', align_corners=False)

        # 调整回原来的形状
        score = output.permute(0, 2, 1)  # 变回 (1, 400, 256)

        # score = self.linear(score.permute(0,2,1)).permute(0,2,1)
        # sc = self.sc_alpha * torch.exp(-0.5 * (1 - score).pow(2) / (self.sc_sigma ** 2)).masked_fill(joint_context_mask, 0)
        # sh[:, :, 0] = sc[:, :, 0] = 1
        # sh, sc = score.masked_fill(joint_head_mask, 0), score.masked_fill(joint_context_mask, 0)
        # sh, sc = sh / (torch.max(sh, dim=1)[0].unsqueeze(1) + 1e-8), \
        #          sc / (torch.max(sc, dim=1)[0].unsqueeze(1) + 1e-8)
        # sh = self.sigmoid(score).masked_fill(joint_head_mask, 0)
        # sc = self.sigmoid(score).masked_fill(joint_context_mask, 0)
        return score.transpose(1, 2), txt_emb


def build_score_module(args):
    return score_module(d_model=args.hidden_dim,
                        dropout=args.dropout,
                        nhead=args.nheads,
                        alpha=args.alpha,
                        sigma=args.sigma)


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
    cosine_dist = cosine_sim
    return cosine_dist


