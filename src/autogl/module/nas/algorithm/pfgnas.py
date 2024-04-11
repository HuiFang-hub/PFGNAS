# Modified from NNI

import logging

import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F

from . import register_nas_algo
from .base import BaseNAS
from ..estimator.base import BaseEstimator
from ..space import BaseSpace
from ..utils import replace_layer_choice, replace_input_choice
# from nni.retiarii.oneshot.pytorch.darts import DartsLayerChoice, DartsInputChoice

_logger = logging.getLogger(__name__)

# copy from nni2.1 for stablility
class PFGNASLayerChoice(nn.Module):
    def __init__(self, layer_choice,device,choose_letter=None):
        super(PFGNASLayerChoice, self).__init__()
        self.name = layer_choice.key
        self.op_choices = nn.ModuleDict(layer_choice.named_children())
        # self.alpha = nn.Parameter(torch.randn(len(self.op_choices)) * 1e-3)
         # operations: 'abd'
        if choose_letter.isdigit(): # choose_letter is None: 1.sigmoid, 2.tanh, 3.relu, 4.linear, 5. elu
            pos = int(choose_letter)
        elif choose_letter.isalpha():
            pos = ord(choose_letter) - ord('a')+1
        else:
            pos = 1
        para_val = generate_list(n=len(self.op_choices), pos=pos)
        self.alpha = torch.tensor(para_val).to(device)
        # self.alpha = nn.Parameter(torch.tensor(para_val))

    def forward(self, *args, **kwargs):
        op_results = torch.stack([op(*args, **kwargs) for op in self.op_choices.values()])
        result = op_results * self.alpha[:, None, None]
        return torch.sum(result, 0)

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self):
        for name, p in super(PFGNASLayerChoice, self).named_parameters():
            if name == 'alpha':
                continue
            yield name, p

    def export(self):
        return torch.argmax(self.alpha).item()


class PFGNASInputChoice(nn.Module):
    def __init__(self, input_choice):
        super(PFGNASInputChoice, self).__init__()
        self.name = input_choice.key
        self.alpha = nn.Parameter(torch.randn(input_choice.n_candidates) * 1e-3)
        self.n_chosen = input_choice.n_chosen or 1

    def forward(self, inputs):
        inputs = torch.stack(inputs)
        alpha_shape = [-1] + [1] * (len(inputs.size()) - 1)
        return torch.sum(inputs * F.softmax(self.alpha, -1).view(*alpha_shape), 0)

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self):
        for name, p in super(PFGNASInputChoice, self).named_parameters():
            if name == 'alpha':
                continue
            yield name, p

    def export(self):
        return torch.argsort(-self.alpha).cpu().numpy().tolist()[:self.n_chosen]


def generate_list(n, pos):
    if not (1 <= pos <= n):
        raise ValueError("Invalid values for n and m")

    # 初始化列表，全部为 0
    result_list = [0] * n

    # 在第 m 个位置上放置 0.99
    result_list[pos - 1] = 0.99

    # 计算其他位置的值，使得列表之和为 1.000
    remaining_sum = 1.000 - 0.99
    for i in range(n):
        if i != pos - 1:
            result_list[i] = remaining_sum / (n - 1)

    return result_list


class pf_serve_LayerChoice(nn.Module):
    def __init__(self, layer_choice,device):
        super(pf_serve_LayerChoice, self).__init__()
        self.name = layer_choice.key
        self.op_choices = nn.ModuleDict(layer_choice.named_children())
        # self.alpha = nn.Parameter(torch.randn(len(self.op_choices)) * 1e-3)
        self.alpha =  (torch.randn(len(self.op_choices)) * 1e-3).to(device)

    def forward(self, *args, **kwargs):
        op_results = torch.stack([op(*args, **kwargs) for op in self.op_choices.values()])
        alpha_shape = [-1] + [1] * (len(op_results.size()) - 1)
        return torch.sum(op_results * F.softmax(self.alpha, -1).view(*alpha_shape), 0)

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self):
        for name, p in super(pf_serve_LayerChoice, self).named_parameters():
            if name == 'alpha':
                continue
            yield name, p

    def export(self):
        return torch.argmax(self.alpha).item()