# -*- coding: utf-8 -*-
# @Time    : 2023/10/31 16:45
# @Function:
import torch_geometric.nn as gnn
import torch.nn as nn

class SGC(nn.Module):
    '''
    r"""The simple graph convolutional operator from the `"Simplifying Graph
    Convolutional Networks" <https://arxiv.org/abs/1902.07153>`_ paper

    '''
    def __init__(self, in_channels,out_channels):
        super(SGC, self).__init__()
        self.sgc = gnn.SGConv(in_channels,out_channels, 2, False)
        return

    def forward(self, x, edge_index):
        x = self.sgc(x, edge_index)
        return x