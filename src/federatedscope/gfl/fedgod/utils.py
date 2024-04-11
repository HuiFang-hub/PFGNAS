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

def generate_rw_subgraph(pyg_graph, nb_nodes, subgraph_size):
    """Generate subgraph with random walk algorithm."""
    row, col = pyg_graph.edge_index
    all_idx = torch.tensor(list(range(nb_nodes)))
    traces = random_walk(row, col, all_idx, walk_length=subgraph_size-1)
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

