import math
import torch
from torch import nn
from torch.autograd import Variable

from utils.misc import NestedTensor


@torch.no_grad()
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数列
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数列
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        bs = x.shape[0]
        pe = Variable(self.pe, requires_grad=False)
        pe = pe.repeat(bs, 1, 1)
        return pe


def build_position_encoding(hidden_dim, max_query_len):
    position_embedding = PositionalEncoding(hidden_dim, max_query_len)
    return position_embedding
