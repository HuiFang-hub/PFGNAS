from torch._C import Graph
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch_geometric.nn as gnn
import src.FLAGNNS.utils as utils
from torch import optim
from torch.autograd import Variable
import pickle
import numpy as np

import warnings
warnings.filterwarnings("ignore")


HIDDENSIZE = 32
INPUT_SIZE = HIDDENSIZE
LAYERS = 1
LR_GRAPHNAS = 0.00035
EPOCHS = 200


class Identity(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x, x1):
        return x


class GMMConv(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """

    def __init__(self, insize, outsize, d, K, *args):
        super(GMMConv, self).__init__()
        self.gmmconv = gnn.GMMConv(insize, outsize, d, K)
        # self.edge_attr = None
        return

    def forward(self, x, edge_index):
        # if self.edge_attr == None:
        edge_attr = utils.cal_edge_attr_for_gmmconv(edge_index)
        # print(x.size(),edge_index.size(),edge_attr.size())
        # print('in gmm conv,type of machine:{}{}{}'.format(x.device,edge_index.device,edge_attr.device))
        return self.gmmconv(x, edge_index, edge_attr)


class Gat(nn.Module):
    def __init__(self, nfeat, nclass):
        super(Gat, self).__init__()
        self.gat1 = gnn.GATConv(nfeat, 8, 8, True)
        self.gat2 = gnn.GATConv(64, nclass)
        return

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        return x


class Sage(nn.Module):
    def __init__(self, nfeat, nclass):
        super(Sage, self).__init__()
        self.sage1 = gnn.SAGEConv(nfeat, 64)
        self.sage2 = gnn.SAGEConv(64, nclass)
        return

    def forward(self, x, edge_index):
        x = self.sage1(x, edge_index)
        x = F.sigmoid(x)
        x = self.sage2(x, edge_index)
        return x


class Gcn(nn.Module):
    def __init__(self, nfeat, nclass):
        super(Gcn, self).__init__()
        self.gcn1 = gnn.GCNConv(nfeat, 64)
        self.gcn2 = gnn.GCNConv(64, nclass)
        return

    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index)
        x = F.relu(x)
        x = self.gcn2(x, edge_index)
        return x


class Sgc(nn.Module):
    def __init__(self, nfeat, nclass):
        super(Sgc, self).__init__()
        self.sgc = gnn.SGConv(nfeat, nclass, 2, False)
        return

    def forward(self, x, edge_index):
        x = self.sgc(x, edge_index)
        return x


class Appnp(nn.Module):
    def __init__(self, nfeat, nclass):
        super(Appnp, self).__init__()
        self.linear = nn.Linear(nfeat, 64)
        self.linear2 = nn.Linear(64, nclass)
        self.appnp = gnn.APPNP(K=10, alpha=0.1)
        return

    def forward(self, x, edge_index):
        x = F.relu(self.linear(x))
        x = self.linear2(x)
        x = self.appnp(x, edge_index)
        return x


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


class Arma(nn.Module):
    def __init__(self, nfeat, nclass):
        super(Arma, self).__init__()
        self.arma1 = gnn.ARMAConv(nfeat, 16, num_stacks=2)
        self.arma2 = gnn.ARMAConv(16, nclass, num_stacks=2)
        return

    def forward(self, x, edge_index):
        x = self.arma1(x, edge_index)
        x = self.arma2(x, edge_index)
        return x


class Structure():
    def __init__(self):
        return

    def set_y(self):
        self.y_1 = 'Gat(self.len, self.len)'  # 6:755
        self.y_2_local_len = 64
        self.y_2 = 'gnn.GINConv(nn.Sequential({},{},{},{}),train_eps =True)'\
            .format('nn.Linear(self.len,self.y_2_local_len)',
                    'nn.BatchNorm1d(self.y_2_local_len)',
                    'nn.ReLU()',
                    'nn.Linear(self.y_2_local_len, self.len)')  # 729
        self.y_3 = 'Sage(self.len, self.len)'  # 714
        self.y_4 = 'Gcn(self.len, self.len)'  # 683
        self.y_5 = 'Sgc(self.len, self.len)'  # 77,759
        self.y_6 = 'Appnp(self.len, self.len)'  # 781
        self.y_7 = 'Agnn(self.len, self.len)'  # 734,686
        self.y_8 = 'Arma(self.len, self.len)'  # 612,556
        self.y_9 = 'gnn.FeaStConv(self.len, self.len,2)'  # 2:728,693
        self.y_10 = 'gnn.GENConv(self.len, self.len)'  # 722,67
        self.y_11 = 'GMMConv(self.len, self.len,2,4)'  # 59,676
        self.y_12 = 'gnn.GatedGraphConv(self.len,3)'
        return


class SonNet(nn.Module):
    def __init__(self, supermask,nfeat, nclass):
        # path = 'tmp.pkl'
        # with open(path, 'rb') as f:
        #     self.supermask = pickle.load(f)
        super(SonNet, self).__init__()
        self.supermask = supermask
        self.len = 64

        self.x_blink = nn.Identity()
        self.z_blink = nn.Identity()

        self.x1 = 'nn.Linear(nfeat, self.len)'
        self.x2 = 'nn.Linear(nfeat, self.len)'
        self.x3 = 'nn.Linear(nfeat, self.len)'
        self.x4 = 'nn.Linear(nfeat, self.len)'
        self.x5 = 'nn.Linear(nfeat, self.len)'
        self.x_1 = 'F.sigmoid'
        self.x_2 = 'F.tanh'
        self.x_3 = 'F.relu'
        self.x_4 = 'F.softmax'
        self.x_5 = 'nn.Identity()'

        self.y_1 = 'gnn.GATConv(self.len, self.len,6,False)'  # 6:755
        y_2_local_len = 64
        self.y_2 = 'gnn.GINConv(nn.Sequential({},{},{},{}),train_eps =True)' \
            .format('nn.Linear(self.len,y_2_local_len)',
                    'nn.BatchNorm1d(y_2_local_len)',
                    'nn.ReLU()',
                    'nn.Linear(y_2_local_len, self.len)')  # 729
        self.y_3 = 'gnn.SAGEConv(self.len, self.len)'  # 714
        self.y_4 = 'gnn.GCNConv(self.len, self.len)'  # 683
        self.y_5 = 'gnn.SGConv(self.len, self.len, 2, False)'  # 77,759
        self.y_6 = 'gnn.APPNP(K=10,alpha=0.1)'  # 781# TODO
        self.y_7 = 'gnn.AGNNConv()'  # 734,686
        self.y_8 = 'gnn.ARMAConv(self.len, self.len,num_stacks =2)'  # 612,556
        self.y_9 = 'gnn.FeaStConv(self.len, self.len,2)'  # 2:728,693
        self.y_10 = 'gnn.GENConv(self.len, self.len)'  # 722,67
        self.y_11 = 'GMMConv(self.len, self.len,2,4)'  # 59,676
        self.y_12 = 'gnn.GatedGraphConv(self.len,3)'

        self.z1 = 'nn.Linear(self.len, nclass)'
        self.z2 = 'nn.Linear(self.len, nclass)'
        self.z3 = 'nn.Linear(self.len, nclass)'
        self.z4 = 'nn.Linear(self.len, nclass)'
        self.z5 = 'nn.Linear(self.len, nclass)'
        self.z_1 = 'F.sigmoid'
        self.z_2 = 'F.tanh'
        self.z_3 = 'F.relu'
        self.z_4 = 'F.softmax'
        self.z_5 = 'nn.Identity()'

        exec('self.x{}=eval(self.x{})'.format(
            self.supermask[0], self.supermask[0]))
        exec('self.x_{}=eval(self.x_{})'.format(
            self.supermask[0], self.supermask[0]))
        exec('self.z{}=eval(self.z{})'.format(
            self.supermask[7], self.supermask[7]))
        exec('self.z_{}=eval(self.z_{})'.format(
            self.supermask[7], self.supermask[7]))

        for i in range(1, 6 + 1):
            tmp_mask = self.supermask[i]
            if tmp_mask == 0:
                break
            layer = 'self.y{}{}_{}'.format(
                int((tmp_mask - 1) / 12), i, (tmp_mask - 1) % 12 + 1)
            exec("{}=eval(self.y_{})".format(layer, (tmp_mask - 1) % 12 + 1))
        return

    def forward(self, x, edge_index):
        supermask = self.supermask
        x = self.x_blink(x) if supermask[0] == 0 else eval('self.x_{}'.format(supermask[0]))(
            eval('self.x{}'.format(supermask[0]))(x))
        if supermask[1] == 0:
            return self.z_blink(x) if supermask[7] == 0 else eval('self.z_{}'.format(supermask[7]))(
                eval('self.z{}'.format(supermask[7]))(x))

        l1_input = x
        l1_output = eval('self.y01_{}'.format(
            supermask[1]))(l1_input, edge_index)
        if supermask[2] == 0:
            x = l1_output
            return self.z_blink(x) if supermask[7] == 0 else eval('self.z_{}'.format(supermask[7]))(
                eval('self.z{}'.format(supermask[7]))(x))

        l2_input = x if (1 <= supermask[2]
                         and supermask[2] <= 12) else l1_output
        l2_output = eval('self.y{}2_{}'.format(int((supermask[2] - 1) / 12), (supermask[2] - 1) % 12 + 1))(l2_input,
                                                                                                           edge_index)
        if supermask[3] == 0:
            x = (l1_output + l2_output) / 2
            return self.z_blink(x) if supermask[7] == 0 else eval('self.z_{}'.format(supermask[7]))(
                eval('self.z{}'.format(supermask[7]))(x))

        l3_input = x if (1 <= supermask[3] and supermask[3] <= 12) else (
            l1_output if (13 <= supermask[3] and supermask[3] <= 24) else l2_output)
        l3_output = eval('self.y{}3_{}'.format(int((supermask[3] - 1) / 12), (supermask[3] - 1) % 12 + 1))(l3_input,
                                                                                                           edge_index)
        if supermask[4] == 0:
            x = (l1_output + l2_output + l3_output) / 3
            return self.z_blink(x) if supermask[7] == 0 else eval('self.z_{}'.format(supermask[7]))(
                eval('self.z{}'.format(supermask[7]))(x))

        l4_input = x if (supermask[4] in [1, 12]) else (l1_output if (supermask[4] in [12 + 1, 12 * 2]) else (
            l2_output if (supermask[4] in [24 + 1, 12 * 3]) else l3_output))
        l4_output = eval('self.y{}4_{}'.format(int((supermask[4] - 1) / 12), (supermask[4] - 1) % 12 + 1))(l4_input,
                                                                                                           edge_index)
        if supermask[5] == 0:
            x = (l1_output + l2_output + l3_output + l4_output) / 4
            return self.z_blink(x) if supermask[7] == 0 else eval('self.z_{}'.format(supermask[7]))(
                eval('self.z{}'.format(supermask[7]))(x))

        l5_input = x if (supermask[5] in [1, 12]) else (l1_output if (supermask[5] in [12 + 1, 12 * 2]) else (
            l2_output if (supermask[5] in [24 + 1, 12 * 3]) else (
                l3_output if (supermask[5] in [36 + 1, 12 * 4]) else l4_output)))
        l5_output = eval('self.y{}5_{}'.format(int((supermask[5] - 1) / 12), (supermask[5] - 1) % 12 + 1))(l5_input,
                                                                                                           edge_index)
        if supermask[6] == 0:
            x = (l1_output + l2_output + l3_output + l4_output + l5_output) / 5
            return self.z_blink(x) if supermask[7] == 0 else eval('self.z_{}'.format(supermask[7]))(
                eval('self.z{}'.format(supermask[7]))(x))

        l6_input = x if (supermask[6] in [1, 12]) else (l1_output if (supermask[6] in [12 + 1, 12 * 2]) else (
            l2_output if (supermask[6] in [24 + 1, 12 * 3]) else (l3_output if (supermask[6] in [36 + 1, 12 * 4]) else (
                l4_output if (supermask[6] in [48 + 1, 12 * 5]) else l5_output))))
        l6_output = eval('self.y{}6_{}'.format(int((supermask[6] - 1) / 12), (supermask[6] - 1) % 12 + 1))(l6_input,
                                                                                                           edge_index)
        x = (l1_output + l2_output + l3_output +
             l4_output + l5_output + l6_output) / 6

        return self.z_blink(x) if supermask[7] == 0 else eval('self.z_{}'.format(supermask[7]))(
            eval('self.z{}'.format(supermask[7]))(x))


class DynamicSonNet(nn.Module, Structure):
    def __init__(self, nfeat, nclass):
        self.supermask = [2, 5, 19, 0, 0, 42, 28, 5]

        super(DynamicSonNet, self).__init__()
        self.len = 64

        self.x_blink = nn.Identity()
        self.z_blink = nn.Identity()

        self.x1 = 'nn.Linear(nfeat, self.len)'
        self.x2 = 'nn.Linear(nfeat, self.len)'
        self.x3 = 'nn.Linear(nfeat, self.len)'
        self.x4 = 'nn.Linear(nfeat, self.len)'
        self.x5 = 'nn.Linear(nfeat, self.len)'
        self.x_1 = 'F.sigmoid'
        self.x_2 = 'F.tanh'
        self.x_3 = 'F.relu'
        self.x_4 = 'F.softmax'
        self.x_5 = 'nn.Identity()'

        Structure.set_y(self)

        self.z1 = 'nn.Linear(self.len, nclass)'
        self.z2 = 'nn.Linear(self.len, nclass)'
        self.z3 = 'nn.Linear(self.len, nclass)'
        self.z4 = 'nn.Linear(self.len, nclass)'
        self.z5 = 'nn.Linear(self.len, nclass)'
        self.z_1 = 'F.sigmoid'
        self.z_2 = 'F.tanh'
        self.z_3 = 'F.relu'
        self.z_4 = 'F.softmax'
        self.z_5 = 'nn.Identity()'

        exec('self.x{}=eval(self.x{})'.format(
            self.supermask[0], self.supermask[0]))
        exec('self.x_{}=eval(self.x_{})'.format(
            self.supermask[0], self.supermask[0]))
        exec('self.z{}=eval(self.z{})'.format(
            self.supermask[7], self.supermask[7]))
        exec('self.z_{}=eval(self.z_{})'.format(
            self.supermask[7], self.supermask[7]))

        for i in range(1, 6 + 1):
            tmp_mask = self.supermask[i]
            if tmp_mask == 0:
                break
            layer = 'self.y{}{}_{}'.format(
                int((tmp_mask - 1) / 12), i, (tmp_mask - 1) % 12 + 1)
            exec("{}=eval(self.y_{})".format(layer, (tmp_mask - 1) % 12 + 1))
        return

    def forward(self, x, edge_index):
        supermask = self.supermask
        x = self.x_blink(x) if supermask[0] == 0 else eval('self.x_{}'.format(supermask[0]))(
            eval('self.x{}'.format(supermask[0]))(x))
        if supermask[1] == 0:
            return self.z_blink(x) if supermask[7] == 0 else eval('self.z_{}'.format(supermask[7]))(
                eval('self.z{}'.format(supermask[7]))(x))

        l1_input = x
        # print('self.y01_{}'.format(supermask[1]))
        l1_output = eval('self.y01_{}'.format(
            supermask[1]))(l1_input, edge_index)
        if supermask[2] == 0:
            x = l1_output
            return self.z_blink(x) if supermask[7] == 0 else eval('self.z_{}'.format(supermask[7]))(
                eval('self.z{}'.format(supermask[7]))(x))

        l2_input = x if (1 <= supermask[2]
                         and supermask[2] <= 12) else l1_output
        # print('self.y{}2_{}'.format(int((supermask[2] - 1)/12),(supermask[2] - 1) % 12 + 1))
        l2_output = eval('self.y{}2_{}'.format(int((supermask[2] - 1) / 12), (supermask[2] - 1) % 12 + 1))(l2_input,
                                                                                                           edge_index)
        if supermask[3] == 0:
            x = (l1_output + l2_output) / 2
            return self.z_blink(x) if supermask[7] == 0 else eval('self.z_{}'.format(supermask[7]))(
                eval('self.z{}'.format(supermask[7]))(x))

        l3_input = x if (1 <= supermask[3] and supermask[3] <= 12) else (
            l1_output if (13 <= supermask[3] and supermask[3] <= 24) else l2_output)
        # print('self.y{}3_{}'.format(int((supermask[3] - 1)/12),(supermask[3] - 1) % 12 + 1))
        l3_output = eval('self.y{}3_{}'.format(int((supermask[3] - 1) / 12), (supermask[3] - 1) % 12 + 1))(l3_input,
                                                                                                           edge_index)
        if supermask[4] == 0:
            x = (l1_output + l2_output + l3_output) / 3
            return self.z_blink(x) if supermask[7] == 0 else eval('self.z_{}'.format(supermask[7]))(
                eval('self.z{}'.format(supermask[7]))(x))

        l4_input = x if (supermask[4] in [1, 12]) else (l1_output if (supermask[4] in [12 + 1, 12 * 2]) else (
            l2_output if (supermask[4] in [24 + 1, 12 * 3]) else l3_output))
        # print('self.y{}4_{}'.format(int((supermask[4] - 1)/12),(supermask[4] - 1) % 12 + 1))
        l4_output = eval('self.y{}4_{}'.format(int((supermask[4] - 1) / 12), (supermask[4] - 1) % 12 + 1))(l4_input,
                                                                                                           edge_index)
        if supermask[5] == 0:
            x = (l1_output + l2_output + l3_output + l4_output) / 4
            return self.z_blink(x) if supermask[7] == 0 else eval('self.z_{}'.format(supermask[7]))(
                eval('self.z{}'.format(supermask[7]))(x))

        l5_input = x if (supermask[5] in [1, 12]) else (l1_output if (supermask[5] in [12 + 1, 12 * 2]) else (
            l2_output if (supermask[5] in [24 + 1, 12 * 3]) else (
                l3_output if (supermask[5] in [36 + 1, 12 * 4]) else l4_output)))
        # print('self.y{}5_{}'.format(int((supermask[5] - 1)/12),(supermask[5] - 1) % 12 + 1))
        l5_output = eval('self.y{}5_{}'.format(int((supermask[5] - 1) / 12), (supermask[5] - 1) % 12 + 1))(l5_input,
                                                                                                           edge_index)
        if supermask[6] == 0:
            x = (l1_output + l2_output + l3_output + l4_output + l5_output) / 5
            return self.z_blink(x) if supermask[7] == 0 else eval('self.z_{}'.format(supermask[7]))(
                eval('self.z{}'.format(supermask[7]))(x))

        l6_input = x if (supermask[6] in [1, 12]) else (l1_output if (supermask[6] in [12 + 1, 12 * 2]) else (
            l2_output if (supermask[6] in [24 + 1, 12 * 3]) else (l3_output if (supermask[6] in [36 + 1, 12 * 4]) else (
                l4_output if (supermask[6] in [48 + 1, 12 * 5]) else l5_output))))
        # print('self.y{}6_{}'.format(int((supermask[6] - 1)/12),(supermask[6] - 1) % 12 + 1))
        l6_output = eval('self.y{}6_{}'.format(int((supermask[6] - 1) / 12), (supermask[6] - 1) % 12 + 1))(l6_input,
                                                                                                           edge_index)
        # print('use 6 nodes')
        x = (l1_output + l2_output + l3_output +
             l4_output + l5_output + l6_output) / 6

        return self.z_blink(x) if supermask[7] == 0 else eval('self.z_{}'.format(supermask[7]))(
            eval('self.z{}'.format(supermask[7]))(x))


class SuperNet(nn.Module, Structure):
    def __init__(self, nfeat, nclass,supermask):
        super(SuperNet, self).__init__()
        self.len = 64
        self.x_blink = nn.Identity()
        self.z_blink = nn.Identity()

        self.x1 = nn.Linear(nfeat, self.len)
        self.x2 = nn.Linear(nfeat, self.len)
        self.x3 = nn.Linear(nfeat, self.len)
        self.x4 = nn.Linear(nfeat, self.len)
        self.x5 = nn.Linear(nfeat, self.len)
        self.x_1 = F.sigmoid
        self.x_2 = F.tanh
        self.x_3 = F.relu
        self.x_4 = F.softmax
        self.x_5 = nn.Identity()

        Structure.set_y(self)

        for j in range(1, 6 + 1):
            for i in range(0, j):
                for t in range(1, 12 + 1):
                    layer = 'self.y{}{}_{}'.format(i, j, t)
                    exec("{}=eval(self.y_{})".format(layer, t))

        self.z1 = nn.Linear(self.len, nclass)
        self.z2 = nn.Linear(self.len, nclass)
        self.z3 = nn.Linear(self.len, nclass)
        self.z4 = nn.Linear(self.len, nclass)
        self.z5 = nn.Linear(self.len, nclass)
        self.z_1 = F.sigmoid
        self.z_2 = F.tanh
        self.z_3 = F.relu
        self.z_4 = F.softmax
        self.z_5 = nn.Identity()
        self.supermask = supermask
        return

    def forward(self, x, edge_index):
        supermask = self.supermask
        x = self.x_blink(x) if supermask[0] == 0 else eval(
            'self.x_{}'.format(supermask[0]))(eval('self.x{}'.format(supermask[0]))(x))
        if supermask[1] == 0:
            return self.z_blink(x) if supermask[7] == 0 else eval('self.z_{}'.format(supermask[7]))(
                eval('self.z{}'.format(supermask[7]))(x))

        l1_input = x
        l1_output = eval('self.y01_{}'.format(
            supermask[1]))(l1_input, edge_index)
        if supermask[2] == 0:
            x = l1_output
            return self.z_blink(x) if supermask[7] == 0 else eval('self.z_{}'.format(supermask[7]))(
                eval('self.z{}'.format(supermask[7]))(x))

        l2_input = x if (1 <= supermask[2]
                         and supermask[2] <= 12) else l1_output
        l2_output = eval('self.y{}2_{}'.format(
            int((supermask[2] - 1)/12), (supermask[2] - 1) % 12 + 1))(l2_input, edge_index)
        if supermask[3] == 0:
            x = (l1_output+l2_output)/2
            return self.z_blink(x) if supermask[7] == 0 else eval('self.z_{}'.format(supermask[7]))(
                eval('self.z{}'.format(supermask[7]))(x))

        l3_input = x if (1 <= supermask[3] and supermask[3] <= 12) else (
            l1_output if (13 <= supermask[3] and supermask[3] <= 24) else l2_output)
        l3_output = eval('self.y{}3_{}'.format(
            int((supermask[3] - 1)/12), (supermask[3] - 1) % 12 + 1))(l3_input, edge_index)
        if supermask[4] == 0:
            x = (l1_output+l2_output+l3_output)/3
            return self.z_blink(x) if supermask[7] == 0 else eval('self.z_{}'.format(supermask[7]))(
                eval('self.z{}'.format(supermask[7]))(x))

        l4_input = x if (supermask[4] in [1, 12]) else (l1_output if (supermask[4] in [
            12+1, 12*2]) else (l2_output if (supermask[4] in [24+1, 12*3]) else l3_output))
        l4_output = eval('self.y{}4_{}'.format(
            int((supermask[4] - 1)/12), (supermask[4] - 1) % 12 + 1))(l4_input, edge_index)
        if supermask[5] == 0:
            x = (l1_output+l2_output+l3_output+l4_output)/4
            return self.z_blink(x) if supermask[7] == 0 else eval('self.z_{}'.format(supermask[7]))(
                eval('self.z{}'.format(supermask[7]))(x))

        l5_input = x if (supermask[5] in [1, 12]) else (l1_output if (supermask[5] in [12+1, 12*2]) else (
            l2_output if (supermask[5] in [24+1, 12*3]) else (l3_output if (supermask[5] in [36+1, 12*4]) else l4_output)))
        l5_output = eval('self.y{}5_{}'.format(
            int((supermask[5] - 1)/12), (supermask[5] - 1) % 12 + 1))(l5_input, edge_index)
        if supermask[6] == 0:
            x = (l1_output+l2_output+l3_output+l4_output+l5_output)/5
            return self.z_blink(x) if supermask[7] == 0 else eval('self.z_{}'.format(supermask[7]))(
                eval('self.z{}'.format(supermask[7]))(x))

        l6_input = x if (supermask[6] in [1, 12]) else (l1_output if (supermask[6] in [12+1, 12*2]) else (l2_output if (supermask[6] in [
            24+1, 12*3]) else (l3_output if (supermask[6] in [36+1, 12*4]) else (l4_output if (supermask[6] in [48+1, 12*5]) else l5_output))))
        l6_output = eval('self.y{}6_{}'.format(
            int((supermask[6] - 1)/12), (supermask[6] - 1) % 12 + 1))(l6_input, edge_index)
        x = (l1_output + l2_output + l3_output +
             l4_output + l5_output + l6_output) / 6

        return self.z_blink(x) if supermask[7] == 0 else eval('self.z_{}'.format(supermask[7]))(eval('self.z{}'.format(supermask[7]))(x))


class Darts(nn.Module, Structure):
    def __init__(self, nfeat, nclass):
        super(Darts, self).__init__()
        self.len = 64
        self.x_blink = nn.Identity()
        self.z_blink = nn.Identity()

        self.x1 = nn.Linear(nfeat, self.len)
        self.x2 = nn.Linear(nfeat, self.len)
        self.x3 = nn.Linear(nfeat, self.len)
        self.x4 = nn.Linear(nfeat, self.len)
        self.x5 = nn.Linear(nfeat, self.len)
        self.x_1 = F.sigmoid
        self.x_2 = F.tanh
        self.x_3 = F.relu
        self.x_4 = F.softmax
        self.x_5 = nn.Identity()

        Structure.set_y(self)

        for j in range(1, 6 + 1):
            for i in range(0, j):
                for t in range(1, 12 + 1):
                    layer = 'self.y{}{}_{}'.format(i, j, t)
                    exec("{}=eval(self.y_{})".format(layer, t))

        self.z1 = nn.Linear(self.len, nclass)
        self.z2 = nn.Linear(self.len, nclass)
        self.z3 = nn.Linear(self.len, nclass)
        self.z4 = nn.Linear(self.len, nclass)
        self.z5 = nn.Linear(self.len, nclass)
        self.z_1 = F.sigmoid
        self.z_2 = F.tanh
        self.z_3 = F.relu
        self.z_4 = F.softmax
        self.z_5 = nn.Identity()

        self.tau = 10
        self.register_parameter('alpha', nn.Parameter(
            Variable(torch.ones(5)/2, requires_grad=True)))
        self.register_parameter('gamma', nn.Parameter(
            Variable(torch.ones(5)/2, requires_grad=True)))
        for j in range(1, 7):
            self.register_parameter('beta{}'.format(j), nn.Parameter(
                eval('Variable(torch.ones({})/2,requires_grad=True)'.format(j*12+1))))
        return

    def forward(self, x, edge_index):
        device = self.alpha.device
        tmp_lis = []
        tmp_alpha = F.softmax(self.alpha)
        for i in range(5):
            tmp_lis.append(
                eval('tmp_alpha[{}] * self.x_{}(self.x{}(x))'.format(i, i+1, i+1)))
        x = torch.sum(torch.stack(tmp_lis, axis=0), dim=0)
        y0 = x
        for j in range(1, 6 + 1):
            tmp_lis = []
            exec('tmp_beta{} = torch.rand(j*12+1)'.format(j))
            for i in range(j):
                exec('tmp_beta{}[{}:{}] = F.softmax(self.beta{}[{}:{}])'.format(
                    j, i*12+1, i*12+13, j, i*12+1, i*12+13))
            for i in range(1, j*12+1):
                tmp_lis.append(eval('self.y{}{}_{}(y{},edge_index)*tmp_beta{}[i]'.format(
                    int((i-1)/12), j, (i-1) % 12+1, int((i-1)/12), j)))
            exec(
                'y{} = torch.sum(torch.stack(tmp_lis,axis=0),dim=0) + 0*self.beta{}[0]**2'.format(j, j))
        x = eval('y1') + eval('y2') + eval('y3') + \
            eval('y4') + eval('y5') + eval('y6')
        tmp_lis = []
        tmp_gamma = F.softmax(self.gamma)
        for i in range(5):
            tmp_lis.append(
                eval('tmp_gamma[{}] * self.z_{}(self.z{}(x))'.format(i, i + 1, i + 1)))
        x = torch.sum(torch.stack(tmp_lis, axis=0), dim=0)
        return x

    def generate_supermask(self):
        device = self.alpha.device
        supermask = [0, 0, 0, 0, 0, 0, 0, 0]
        max_alpha = torch.Tensor([-1000000]).to(device)
        arg_max_alpha = None
        for idx in range(5):
            if eval('torch.abs(self.alpha[{}])'.format(idx)) > max_alpha:
                max_alpha = eval('torch.abs(self.alpha[{}])'.format(idx))
                arg_max_alpha = idx
        supermask[0] = arg_max_alpha + 1

        max_alpha = torch.Tensor([-1000000]).to(device)
        arg_max_alpha = None
        for idx in range(5):
            if eval('torch.abs(self.gamma[{}])'.format(idx)) > max_alpha:
                max_alpha = eval('torch.abs(self.gamma[{}])'.format(idx))
                arg_max_alpha = idx
        supermask[7] = arg_max_alpha + 1

        for j in range(1, 6 + 1):
            max_alpha = torch.Tensor([-1000000]).to(device)
            arg_max_alpha = None
            for i in range(j*12+1):
                if eval('torch.abs(self.beta{}[{}])'.format(j, i)) > max_alpha:
                    max_alpha = eval('torch.abs(self.beta{}[{}])'.format(j, i))
                    arg_max_alpha = i
            supermask[j] = arg_max_alpha
        return supermask


class FedNas(nn.Module, Structure):
    def __init__(self, nfeat, nclass):
        super(FedNas, self).__init__()
        self.len = 64
        self.x_blink = nn.Identity()
        self.z_blink = nn.Identity()

        self.x1 = nn.Linear(nfeat, self.len)
        self.x2 = nn.Linear(nfeat, self.len)
        self.x3 = nn.Linear(nfeat, self.len)
        self.x4 = nn.Linear(nfeat, self.len)
        self.x5 = nn.Linear(nfeat, self.len)
        self.x_1 = F.sigmoid
        self.x_2 = F.tanh
        self.x_3 = F.relu
        self.x_4 = F.softmax
        self.x_5 = nn.Identity()

        Structure.set_y(self)

        for j in range(1, 6 + 1):
            for i in range(0, j):
                for t in range(1, 12 + 1):
                    layer = 'self.y{}{}_{}'.format(i, j, t)
                    exec("{}=eval(self.y_{})".format(layer, t))

        self.z1 = nn.Linear(self.len, nclass)
        self.z2 = nn.Linear(self.len, nclass)
        self.z3 = nn.Linear(self.len, nclass)
        self.z4 = nn.Linear(self.len, nclass)
        self.z5 = nn.Linear(self.len, nclass)
        self.z_1 = F.sigmoid
        self.z_2 = F.tanh
        self.z_3 = F.relu
        self.z_4 = F.softmax
        self.z_5 = nn.Identity()

        self.tau = 10
        self.register_parameter('alpha', nn.Parameter(
            Variable(torch.ones(5)/2, requires_grad=True)))
        self.register_parameter('gamma', nn.Parameter(
            Variable(torch.ones(5)/2, requires_grad=True)))
        for j in range(1, 7):
            self.register_parameter('beta{}'.format(j), nn.Parameter(
                eval('Variable(torch.ones({})/2,requires_grad=True)'.format(j*12+1))))
        return

    def forward(self, x, edge_index):
        device = self.alpha.device
        tmp_lis = []
        tmp_alpha = F.softmax(self.alpha)
        for i in range(5):
            tmp_lis.append(
                eval('tmp_alpha[{}] * self.x_{}(self.x{}(x))'.format(i, i+1, i+1)))
        x = torch.sum(torch.stack(tmp_lis, axis=0), dim=0)
        y0 = x
        for j in range(1, 6 + 1):
            tmp_lis = []
            exec('tmp_beta{} = torch.rand(j*12+1)'.format(j))
            for i in range(j):
                exec('tmp_beta{}[{}:{}] = F.softmax(self.beta{}[{}:{}])'.format(
                    j, i*12+1, i*12+13, j, i*12+1, i*12+13))
            for i in range(1, j*12+1):
                tmp_lis.append(eval('self.y{}{}_{}(y{},edge_index)*tmp_beta{}[i]'.format(
                    int((i-1)/12), j, (i-1) % 12+1, int((i-1)/12), j)))
            exec(
                'y{} = torch.sum(torch.stack(tmp_lis,axis=0),dim=0) + 0*self.beta{}[0]**2'.format(j, j))
        x = eval('y1') + eval('y2') + eval('y3') + \
            eval('y4') + eval('y5') + eval('y6')
        tmp_lis = []
        tmp_gamma = F.softmax(self.gamma)
        for i in range(5):
            tmp_lis.append(
                eval('tmp_gamma[{}] * self.z_{}(self.z{}(x))'.format(i, i + 1, i + 1)))
        x = torch.sum(torch.stack(tmp_lis, axis=0), dim=0)
        return x

    def generate_supermask(self):
        device = self.alpha.device
        supermask = [0, 0, 0, 0, 0, 0, 0, 0]
        max_alpha = torch.Tensor([-1000000]).to(device)
        arg_max_alpha = None
        for idx in range(5):
            if eval('torch.abs(self.alpha[{}])'.format(idx)) > max_alpha:
                max_alpha = eval('torch.abs(self.alpha[{}])'.format(idx))
                arg_max_alpha = idx
        supermask[0] = arg_max_alpha + 1

        max_alpha = torch.Tensor([-1000000]).to(device)
        arg_max_alpha = None
        for idx in range(5):
            if eval('torch.abs(self.gamma[{}])'.format(idx)) > max_alpha:
                max_alpha = eval('torch.abs(self.gamma[{}])'.format(idx))
                arg_max_alpha = idx
        supermask[7] = arg_max_alpha + 1

        for j in range(1, 6 + 1):
            max_alpha = torch.Tensor([-1000000]).to(device)
            arg_max_alpha = None
            for i in range(j*12+1):
                if eval('torch.abs(self.beta{}[{}])'.format(j, i)) > max_alpha:
                    max_alpha = eval('torch.abs(self.beta{}[{}])'.format(j, i))
                    arg_max_alpha = i
            supermask[j] = arg_max_alpha
        return supermask

    def get_parameters(self):
        for name, param in self.named_parameters():
            if name.__contains__("alpha") or name.__contains__("beta") or name.__contains__("gamma"):
                continue
            yield param

    def get_arc_params(self):
        for name, param in self.named_parameters():
            if name.__contains__("alpha") or name.__contains__("beta") or name.__contains__("gamma"):
                yield param


# class GraphNas(nn.Module):
#     def __init__(self, INPUT_SIZE):
#         super(GraphNas, self).__init__()
#         self.rnn = nn.RNN(
#             input_size=INPUT_SIZE,
#             hidden_size=HIDDENSIZE,
#             num_layers=LAYERS,
#             # batch_first=True
#         )

#         self.out1 = nn.Linear(HIDDENSIZE, 5)
#         self.out2 = nn.Linear(HIDDENSIZE, 12)
#         self.out8 = nn.Linear(HIDDENSIZE, 5)
#         for i in range(3, 7 + 1):
#             exec("self.out{} = nn.Linear(HIDDENSIZE, 13)".format(i))

#         self.b = 0
#         self.beta = 0.9
#         return

#     def generate_code(self):
#         h_state = torch.FloatTensor(np.zeros((1, 1, INPUT_SIZE)))
#         if torch.cuda.is_available():
#             h_state = h_state.cuda()
#         else:
#             h_state = h_state.cpu()
#         x = h_state
#         res = []
#         for i in range(1, 8 + 1):
#             r_out, h_state = self.rnn(x, h_state)
#             res.append(
#                 F.softmax(eval("self.out{}".format(i))(r_out[0]), dim=1))
#             x = h_state
#         return res

#     def get_loss(self, dummy_code, R):
#         loss0 = 0
#         idx = 1
#         for code in dummy_code:
#             exec(
#                 "loss{} = loss{} + (-torch.log(torch.max(code))*({} - self.b))".format(idx, idx - 1, R))
#             idx += 1
#         self.b = self.beta * self.b + (1 - self.beta) * R
#         print("reward~R:{},b:{}".format(R, self.b))
#         return eval("loss{}".format(len(dummy_code)))

#     def parse_code(self, dummy_code):
#         supermask = []
#         idx = 1
#         for code in dummy_code:
#             if idx == 1 or idx == 8:
#                 supermask.append(int(torch.argmax(code) + 1))
#             else:
#                 supermask.append(int(torch.argmax(code)))
#             idx += 1
#         return supermask


class GraphNas(nn.Module):
    def __init__(self, input_size,device):
        super(GraphNas, self).__init__()
        self.input_size = input_size
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=HIDDENSIZE,
            num_layers=LAYERS,
            # batch_first=True
        )
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=HIDDENSIZE,
            # num_layers=LAYERS,
        )

        self.out1 = nn.Linear(HIDDENSIZE, 5)
        self.out2 = nn.Linear(HIDDENSIZE, 12)
        self.out8 = nn.Linear(HIDDENSIZE, 5)
        for i in range(3, 7 + 1):
            exec("self.out{} = nn.Linear(HIDDENSIZE, 13)".format(i))

        self.b = 0
        self.beta = 0.9
        self.device = device
        return

    def generate_code(self):
        h_state = [torch.FloatTensor(np.zeros((1, 1, self.input_size))),
                   torch.FloatTensor(np.zeros((1, 1, self.input_size)))]
        if torch.cuda.is_available():
            h_state[0] = h_state[0].to(self.device)
            h_state[1] = h_state[1].to(self.device)
        x = h_state[0]
        res = []
        for i in range(1, 8 + 1):
            r_out, h_state = self.lstm(x, h_state)
            res.append(F.softmax(eval("self.out{}".format(i))(r_out[0]), dim=1))
            x = h_state[0]
        return res

    def parse_code(self, dummy_code):
        supermask = []
        idx = 1
        for code in dummy_code:
            tmp = code.multinomial(num_samples=1).cpu().numpy()
            tmp = tmp[0][0]
            if idx == 1 or idx == 8:
                supermask.append(tmp + 1)
            else:
                supermask.append(tmp)
            idx += 1
        # supermask = dummy_code.multinomial(
        #     num_samples=1).reshape(len(dummy_code))
        # supermask[1:-1] += 1
        return supermask

    def get_loss(self, dummy_code, supermask, R):
        # code = dummy_code[0]
        # loss = torch.log(torch.max(code)) * (R - self.b)
        # loss.backward()
        # loss_sum += loss.item()
        losses = []
        for i in range(len(dummy_code)):
            index = supermask[i]
            if i in [0, 7]:
                index -= 1
            losses.append(torch.log(dummy_code[i][0][index]) * (R - self.b))
        # idx = 1
        # for code in dummy_code:
        #     exec(
        #         "loss{} = loss{} + (torch.log(torch.max(code))*({} - self.b))".format(idx, idx - 1, R))
        #     idx += 1
        self.b = self.beta * self.b + (1 - self.beta) * R
        return torch.stack(losses).mean()


if __name__ == "__main__":
    device = torch.device("cuda:" + str(2))
    model = GraphNas(32,device)
    # model.parameters()
    if torch.cuda.is_available():
        torch.cuda.init()
        model.to(device)
    dummy_code = model.generate_code()
    supermask = model.parse_code(dummy_code)
    print(dummy_code)
    print(supermask)
    pass
