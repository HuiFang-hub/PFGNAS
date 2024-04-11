import torch

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_networkx, from_networkx

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_adj
from torch_cluster import random_walk
from numpy import percentile
from src.federatedscope.core.configs.config import global_cfg
import dgl

class HideGraph(BaseTransform):
    r"""
    Generate impaired graph with labels and features to train NeighGen,
    hide Node from validation set from raw graph.

    Arguments:
        hidden_portion (int): hidden_portion of validation set.
        num_pred (int): hyperparameters which limit
            the maximum value of the prediction

    :returns:
        filled_data : impaired graph with attribute "num_missing"
    :rtype:
        nx.Graph
    """
    def __init__(self, hidden_portion=0.5, num_pred=5):
        self.hidden_portion = hidden_portion
        self.num_pred = num_pred

    def __call__(self, data):

        # 找到验证集的节点编号
        val_ids = torch.where(data.val_mask == True)[0]
        # 从验证集中随机选择一定比例的节点进行隐藏
        hide_ids = np.random.choice(val_ids,
                                    int(len(val_ids) * self.hidden_portion),
                                    replace=False)
        # 新建一个掩码，用来表示哪些节点被保留下来了
        remaining_mask = torch.ones(data.num_nodes, dtype=torch.bool)
        # 被隐藏的节点的掩码变成False
        remaining_mask[hide_ids] = False
        # 找到保留下来的节点编号
        remaining_nodes = torch.where(remaining_mask == True)[0].numpy()

        # 为每个节点添加一个列表，用来存储那些在该节点的邻居中被隐藏的节点的编号
        data.ids_missing = [[] for _ in range(data.num_nodes)]

        # 将DGL图对象转换为NetworkX图对象，便于操作
        if "y" in data:
            node_attrs = [
                            'x', 'y', 'ay','train_mask', 'val_mask', 'test_mask',
                            'index_orig', 'ids_missing'
                        ]
        else:
            node_attrs = [
                'x', 'ay','train_mask', 'val_mask', 'test_mask',
                            'index_orig', 'ids_missing'
            ]

        G = to_networkx(data,node_attrs=node_attrs,to_undirected=True)

        # 遍历所有被隐藏的节点的邻居，并在其邻居节点的节点信息中的ids_missing列表中添加该被隐藏的节点的编号
        for missing_node in hide_ids:
            neighbors = G.neighbors(missing_node)
            for i in neighbors:
                G.nodes[i]['ids_missing'].append(missing_node)
        # 遍历所有节点，处理被隐藏的节点的信息，将它们的ids_missing列表删除，用num_missing字段记录被隐藏节点的数量
        # 用x_missing字段记录被隐藏的节点的特征向量（如果它们的邻居中被隐藏节点的数量小于等于num_pred，则特征向量的长度用0补齐，
        # 否则仅取前num_pred个被隐藏的节点的特征向量）
        for i in G.nodes:
            ids_missing = G.nodes[i]['ids_missing']
            del G.nodes[i]['ids_missing']
            G.nodes[i]['num_missing'] = torch.tensor([len(ids_missing)], dtype=torch.float32) #np.array([len(ids_missing)],dtype=np.float32)
            if len(ids_missing) > 0:
                if len(ids_missing) <= self.num_pred:
                    G.nodes[i]['x_missing'] = torch.cat(
                        (data.x[ids_missing],
                         torch.zeros((self.num_pred - len(ids_missing),data.x.shape[1]))))
                else:
                    G.nodes[i]['x_missing'] = data.x[ids_missing[:self.num_pred]]
            else:
                G.nodes[i]['x_missing'] = torch.zeros(
                    (self.num_pred, data.x.shape[1]))

        # 将处理过的NetworkX图对象转换为DGL图对象并返回
        return from_networkx(nx.subgraph(G, remaining_nodes))

    def __repr__(self):
        return f'{self.__class__.__name__}({self.hidden_portion})'





def FillGraph(impaired_data, original_data, pred_missing, pred_feats,
              num_pred):
    # Mend the original data
    original_data = original_data.detach().cpu()
    new_features = original_data.x
    new_edge_index = original_data.edge_index.T
    pred_missing = pred_missing.detach().cpu().numpy()
    pred_feats = pred_feats.detach().cpu().reshape(
        (-1, num_pred, original_data.num_node_features))

    start_id = original_data.num_nodes
    for node in range(len(pred_missing)):
        num_fill_node = np.around(pred_missing[node]).astype(np.int32).item()
        if num_fill_node > 0:
            new_ids_i = np.arange(start_id,
                                  start_id + min(num_pred, num_fill_node))
            org_id = impaired_data.index_orig[node]
            org_node = torch.where(
                original_data.index_orig == org_id)[0].item()
            new_edges = torch.tensor([[org_node, fill_id]
                                      for fill_id in new_ids_i],
                                     dtype=torch.int64)
            new_features = torch.vstack(
                (new_features, pred_feats[node][:num_fill_node]))
            new_edge_index = torch.vstack((new_edge_index, new_edges))
            start_id = start_id + min(num_pred, num_fill_node)
    new_y = torch.zeros(new_features.shape[0], dtype=torch.int64)
    new_y[:original_data.num_nodes] = original_data.y
    # new_ay = torch.zeros(new_features.shape[0], dtype=torch.int64)
    new_ay = torch.full((new_features.shape[0],), 3, dtype=torch.int64)
    new_ay[:original_data.num_nodes] = original_data.ay
    filled_data = Data(
        x=new_features,
        edge_index=new_edge_index.T,
        train_idx=torch.where(original_data.train_mask == True)[0],
        valid_idx=torch.where(original_data.val_mask == True)[0],
        test_idx=torch.where(original_data.test_mask == True)[0],
        y=new_y,
        ay=new_ay
    )
    return filled_data


@torch.no_grad()
def GraphMender(model, impaired_data, original_data):
    r"""Mend the graph with generation model
    Arguments:
        model (torch.nn.module): trained generation model
        impaired_data (PyG.Data): impaired graph
        original_data (PyG.Data): raw graph
    :returns:
        filled_data : Graph after Data Enhancement
    :rtype:
        PyG.data
    """
    device = impaired_data.x.device
    model = model.to(device)
    pred_missing, pred_feats, _ = model(impaired_data)

    return FillGraph(impaired_data, original_data, pred_missing, pred_feats,
                     global_cfg.fedsageplus.num_pred)


def process_graph(G,device):
    """
    Description
    -----------
    Process the raw PyG data object into a tuple of sub data
    objects needed for the model.

    Parameters
    ----------
    G : PyTorch Geometric Data instance (torch_geometric.data.Data)
        The input data.

    Returns
    -------
    x : torch.Tensor
        Attribute (feature) of nodes.
    adj : torch.Tensor
        Adjacency matrix of the graph.
    """
    num_nodes = G.x.shape[0]
    feat_dim = G.x.shape[1]
    adj = to_dense_adj(G.edge_index)[0]
    adj = sp.coo_matrix(adj.cpu().numpy())  #
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    adj = (adj + sp.eye(adj.shape[0])).todense()

    x = G.x[np.newaxis]
    adj = torch.FloatTensor(adj[np.newaxis]).to(device)
    # return data objects needed for the network

    return  num_nodes, feat_dim, x, adj


class CoLA_Base(nn.Module):
    def __init__(self,
                 n_in,
                 n_h,
                 activation,
                 negsamp_round,
                 readout,
                 subgraph_size,
                 device):

        super(CoLA_Base, self).__init__()
        self.n_in = n_in
        self.subgraph_size = subgraph_size
        self.device = device
        self.readout = readout
        self.gcn = GCN(n_in, n_h, activation).to(self.device)
        self.disc = Discriminator(n_h, negsamp_round)

    def forward(self,  x, adj, idx, subgraphs, batch_size, sparse=False):

        batch_adj = []
        batch_feature = []
        added_adj_zero_row = torch.zeros(
            (batch_size, 1, self.subgraph_size)).to(self.device)
        added_adj_zero_col = torch.zeros(
            (batch_size, self.subgraph_size + 1, 1)).to(self.device)
        added_adj_zero_col[:, -1, :] = 1.
        added_feat_zero_row = torch.zeros((batch_size, 1,
                                           self.n_in)).to(self.device)

        for i in idx:
            cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
            cur_feat = x[:, subgraphs[i], :]
            batch_adj.append(cur_adj)
            batch_feature.append(cur_feat)

        batch_adj = torch.cat(batch_adj)
        batch_adj = torch.cat((batch_adj, added_adj_zero_row), dim=1)
        batch_adj = torch.cat((batch_adj, added_adj_zero_col), dim=2)
        batch_feature = torch.cat(batch_feature)
        batch_feature = torch.cat(
            (batch_feature[:, :-1, :],
             added_feat_zero_row,
             batch_feature[:, -1:, :]), dim=1)

        h_1 = self.gcn(batch_feature, batch_adj, sparse)
        #
        if self.readout == 'max':
            h_mv = h_1[:, -1, :]
            c = torch.max(h_1[:, : -1, :], 1).values
        elif self.readout == 'min':
            h_mv = h_1[:, -1, :]
            c = torch.min(h_1[:, : -1, :], 1).values
        elif self.readout == 'avg':
            h_mv = h_1[:, -1, :]
            c = torch.mean(h_1[:, : -1, :], 1)
        elif self.readout == 'weighted_sum':
            seq, query = h_1[:, : -1, :], h_1[:, -2: -1, :],
            query = query.permute(0, 2, 1)
            sim = torch.matmul(seq, query)
            sim = F.softmax(sim, dim=1)
            sim = sim.repeat(1, 1, 64)
            out = torch.mul(seq, sim)
            c = torch.sum(out, 1)
            h_mv = h_1[:, -1, :]

        ret = self.disc(c, h_mv)

        return ret



class Discriminator(nn.Module):
    def __init__(self, n_h, negsamp_ratio):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

        self.negsamp_ratio = negsamp_ratio

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl):
        scs = []
        # positive
        scs.append(self.f_k(h_pl, c))

        # negative
        c_mi = c
        for _ in range(self.negsamp_ratio):
            c_mi = torch.cat((c_mi[-2:-1, :], c_mi[:-1, :]), 0)
            scs.append(self.f_k(h_pl, c_mi))

        logits = torch.cat(tuple(scs))

        return logits
