# -*- coding: utf-8 -*-
# @Time    : 2023/10/31 16:46
# @Function:
import torch_geometric.nn as gnn
import torch.nn as nn
import sys
import pickle
class ARMA(nn.Module):
    '''
    Graph neural networks with convolutional ARMA filters

    '''
    def __init__(self,  in_channels, out_channels,hidden):
        super(ARMA, self).__init__()
        self.arma1 = gnn.ARMAConv(in_channels, hidden, num_stacks=2)
        self.arma2 = gnn.ARMAConv(hidden,out_channels, num_stacks=2)
        return

    def forward(self, x, edge_index):
        x = self.arma1(x, edge_index)
        x = self.arma2(x, edge_index)
        return x

# model = ARMA(64,7,64)
# model_size = sys.getsizeof(pickle.dumps(
#                 model)) / 1024.0 * 8.
#
# print(model_size)