# -*- coding: utf-8 -*-
# @Time    : 12/04/2023 09:13
# @Function:
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_adj
from torch_cluster import random_walk
import numpy as np
from numpy import percentile

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
    edge_index = G.edge_index
    adj = to_dense_adj(edge_index)[0]
    # adj = adj.cpu().numpy()
    x = G.x
    if hasattr(G, 'y'):
        y = G.y
    else:
        y = None
    # return data objects needed for the network
    num_nodes = x.shape[0]
    feat_dim = x.shape[1]
    adj = normalize_adj(adj)
    adj = (adj + torch.eye(adj.shape[0]))

    x = torch.FloatTensor(x[np.newaxis])
    adj = torch.FloatTensor(adj[np.newaxis])

    edge_index = edge_index.to(device)
    return x, adj, edge_index, y, num_nodes, feat_dim


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    # adj = sp.coo_matrix(adj)  # 将输入的邻接矩阵转换成 Scipy COO 稀疏矩阵
    # rowsum = np.array(adj.sum(1))  # 计算每一行的元素之和，得到一个列向量
    # d_inv_sqrt = np.power(rowsum, -0.5).flatten()  # 每个元素取倒数再开方，并转换成一维数组
    # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.  # 处理无穷大的情况
    # d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # 将一维数组转换成对角线矩阵
    # # 对称标准化邻接矩阵，并将结果表示成 COO 格式
    # return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    n = adj.shape[0]  # 获取邻接矩阵大小
    eye = torch.eye(n).to(adj.device)  # 创建一个单位矩阵并移到与邻接矩阵相同的设备上
    adj = adj + eye  # 将单位矩阵加入邻接矩阵中
    deg_inv_sqrt = torch.pow(adj.sum(dim=1), -0.5)  # 计算度矩阵的逆平方根
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.  # 将无穷大的项设为0
    deg_inv_sqrt_matrix = torch.diag(deg_inv_sqrt)  # 构建度矩阵的逆平方根对角矩阵
    return deg_inv_sqrt_matrix @ adj @ deg_inv_sqrt_matrix #adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).T
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def generate_rw_subgraph(pyg_graph, nb_nodes, subgraph_size):
    """Generate subgraph with random walk algorithm."""
    row, col = pyg_graph.edge_index
    all_idx = torch.tensor(list(range(nb_nodes)))
    traces = random_walk(row, col, all_idx, walk_length=3)
    subv = traces.tolist()
    return subv

def loss_function( logits, batch_size,negsamp_ratio):

    b_xent = nn.BCEWithLogitsLoss(reduction='none',
                                  pos_weight=torch.tensor(
                                      [negsamp_ratio]))

    lbl = torch.unsqueeze(
        torch.cat((torch.ones(batch_size),
                   torch.zeros(batch_size * negsamp_ratio))), 1)

    score = b_xent(logits.cpu(), lbl.cpu())

    return score

def _process_decision_scores(decision_scores_,contamination):
    """Internal function to calculate key attributes:
    - threshold_: used to decide the binary label
    - labels_: binary labels of training data
    Returns
    -------
    self
    """

    threshold_ = percentile(decision_scores_,
                                 100 * (1 - contamination))
    labels_ = (decision_scores_ > threshold_).astype('int').ravel()

    # calculate for predict_proba()

    _mu = np.mean(decision_scores_)
    _sigma = np.std(decision_scores_)

    return

