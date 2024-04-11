# -*- coding: utf-8 -*-
# @Time    : 2023/10/31 17:04
# @Function:
import torch_geometric.nn as gnn
import torch.nn as nn
import torch.nn.functional as F

class APPNP(nn.Module):
    '''
    Predict then propagate: Graph neural networks meet personalized PageRank

    '''
    def __init__(self, in_channels,out_channels,hidden):
        super(APPNP, self).__init__()
        self.linear = nn.Linear(in_channels,hidden)
        self.linear2 = nn.Linear(hidden, out_channels)
        self.appnp = gnn.APPNP(K=10, alpha=0.1)
        return

    def forward(self, x, edge_index):
        x = F.relu(self.linear(x))
        x = self.linear2(x)
        x = self.appnp(x, edge_index)
        return x
