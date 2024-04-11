# -*- coding: utf-8 -*-
# @Time    : 2023/12/14 19:34
# @Function:
import torch
from torch import nn
class LinearConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(LinearConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = torch.nn.Linear(in_channels, out_channels, bias)

    def forward(self, x, edge_index, edge_weight=None):
        return self.linear(x)

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, self.in_channels, self.out_channels
        )