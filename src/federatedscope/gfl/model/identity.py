import torch
from torch import nn

class Identity(nn.Module):
    def forward(self, x, edge_index, edge_weight=None):
        return x

    def __repr__(self):
        return "Identity()"

class ZeroConv(nn.Module):
    def forward(self, x, edge_index, edge_weight=None):
        out = torch.zeros_like(x)
        out.requires_grad = True
        return out

    def __repr__(self):
        return "ZeroConv()"