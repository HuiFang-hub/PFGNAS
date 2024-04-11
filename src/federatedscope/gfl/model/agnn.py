# -*- coding: utf-8 -*-
# @Time    : 2023/10/31 16:47
# @Function:
import torch_geometric.nn as gnn
import torch.nn as nn

class Agnn(nn.Module):
    def __init__(self, nfeat, nclass):
        super(Agnn, self).__init__()
        self.agnn1 = gnn.AGNNConv()
        self.agnn2 = gnn.AGNNConv()
        self.agnn3 = gnn.AGNNConv()
        self.agnn4 = gnn.AGNNConv()
        return

    def forward(self, x, edge_index):
        x = self.agnn1(x, edge_index)
        x = self.agnn2(x, edge_index)
        x = self.agnn3(x, edge_index)
        return x
