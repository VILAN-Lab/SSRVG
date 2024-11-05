import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from utils.misc import NestedTensor, is_main_process


class sampling_for_three(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        sampling_list = []
        conv_layer = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2, stride=2)
        sampling_list.append(conv_layer)
        conv_layer = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1)
        sampling_list.append(conv_layer)
        conv_layer = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
        sampling_list.append(conv_layer)
        self.sampling_list = nn.ModuleList(sampling_list)

    def forward(self, xs):
        new_xs = []
        for i, x in enumerate(xs):
             new_xs.append(self.sampling_list[i](x).flatten(-2))
        return new_xs


class sampling_swin(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        size = [8, 4, 2, 1]
        target_size = 20
        sampling_list = []
        for s in size:
            conv_layer = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=s, stride=s)
            # conv_layer = nn.Sequential(
            #     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=s, stride=s),
            #     nn.BatchNorm2d(hidden_dim),
            #     nn.ReLU()
            # )
            sampling_list.append(conv_layer)
        conv_layer = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
        # conv_layer = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm2d(hidden_dim),
        #     nn.ReLU()
        # )
        sampling_list.append(conv_layer)
        self.sampling_list = nn.ModuleList(sampling_list)

    def forward(self, xs):
        for i, x in enumerate(xs):
            xs[i] = self.sampling_list[i](x).flatten(-2)
        return xs


class sampling(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        dims = [512, 1024, 2048]
        sampling_list = []
        l = len(dims)
        for dim in dims:
            reduc = nn.Sequential(
                nn.Conv2d(dim, hidden_dim, kernel_size=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU()
            )
            layer = nn.Sequential(reduc)
            if l == 1:
                sampling_list.append(layer)
                break
            for i in range(1, l):
                layer.add_module(f'{i}', sampling_layer(hidden_dim))
            sampling_list.append(layer)
            l = l - 1

        self.sampling_list = nn.ModuleList(sampling_list)

    def forward(self, xs):

        for i, x in enumerate(xs):
            xs[i] = self.sampling_list[i](x).flatten(-2)

        return xs


class sampling_layer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.down_sampling = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.down_sampling(x)
