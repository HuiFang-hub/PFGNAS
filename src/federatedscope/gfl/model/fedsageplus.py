from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import random

import torch
import numpy as np
import scipy.sparse as sp

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from pygod.models.basic_nn import Vanilla_GCN as GCN
from src.federatedscope.gfl.model import SAGE_Net
import dgl
from dgl.nn import GATv2Conv
# from src.dgld.models.CoLA import CoLA
"""
https://proceedings.neurips.cc//paper/2021/file/ \
34adeb8e3242824038aa65460a47c29e-Paper.pdf
Fedsageplus models from the "Subgraph Federated Learning with Missing
Neighbor Generation" (FedSage+) paper, in NeurIPS'21
Source: https://github.com/zkhku/fedsage
"""
class Discriminator(torch.nn.Module):
    def __init__(self, n_h, negsamp_ratio):
        super(Discriminator, self).__init__()
        self.f_k = torch.nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

        self.negsamp_ratio = negsamp_ratio

    def weights_init(self, m):
        if isinstance(m, torch.nn.Bilinear):
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


class CoLA_Base(torch.nn.Module):
    def __init__(self,
                 n_in,
                 n_h,
                 activation,
                 negsamp_round,
                 readout,
                 subgraph_size,
                 ):

        super(CoLA_Base, self).__init__()
        self.n_in = n_in
        self.subgraph_size = subgraph_size
        # self.device = device
        self.readout = readout
        self.gcn = GCN(n_in, n_h, activation)
        self.disc = Discriminator(n_h, negsamp_round)

    def forward(self, x, adj, idx, subgraphs, batch_size, sparse=False):

        batch_adj = []
        batch_feature = []
        added_adj_zero_row = torch.zeros(
            (batch_size, 1, self.subgraph_size))
        added_adj_zero_col = torch.zeros(
            (batch_size, self.subgraph_size + 1, 1))
        added_adj_zero_col[:, -1, :] = 1.
        added_feat_zero_row = torch.zeros((batch_size, 1,
                                           self.n_in))

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

class Sampling(nn.Module):
    def __init__(self):
        super(Sampling, self).__init__()

    def forward(self, inputs):
        rand = torch.normal(0, 1, size=inputs.shape)

        return inputs + rand.to(inputs.device)


class FeatGenerator(nn.Module):
    def __init__(self, latent_dim, dropout, num_pred, feat_shape):
        super(FeatGenerator, self).__init__()
        self.num_pred = num_pred
        self.feat_shape = feat_shape
        self.dropout = dropout
        self.sample = Sampling()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 2048)
        self.fc_flat = nn.Linear(2048, self.num_pred * self.feat_shape)

    def forward(self, x):
        x = self.sample(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.tanh(self.fc_flat(x))

        return x


class NumPredictor(nn.Module):
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        super(NumPredictor, self).__init__()
        self.reg_1 = nn.Linear(self.latent_dim, 1)

    def forward(self, x):
        x = F.relu(self.reg_1(x))
        return x


# Mend the graph via NeighGen
class MendGraph(nn.Module):
    def __init__(self, num_pred):
        super(MendGraph, self).__init__()
        self.num_pred = num_pred
        for param in self.parameters():
            param.requires_grad = False

    def mend_graph(self, x, edge_index, pred_degree, gen_feats):
        device = gen_feats.device
        num_node, num_feature = x.shape
        new_edges = []
        gen_feats = gen_feats.view(-1, self.num_pred, num_feature)

        if pred_degree.device.type != 'cpu':
            pred_degree = pred_degree.cpu()
        pred_degree = torch._cast_Int(torch.round(pred_degree)).detach()
        x = x.detach()
        fill_feats = torch.vstack((x, gen_feats.view(-1, num_feature)))

        for i in range(num_node):
            for j in range(min(self.num_pred, max(0, pred_degree[i]))):
                new_edges.append(
                    np.asarray([i, num_node + i * self.num_pred + j]))

        new_edges = torch.tensor(np.asarray(new_edges).reshape((-1, 2)),
                                 dtype=torch.int64).T
        new_edges = new_edges.to(device)
        if len(new_edges) > 0:
            fill_edges = torch.hstack((edge_index, new_edges))
        else:
            fill_edges = torch.clone(edge_index)
        return fill_feats, fill_edges

    def forward(self, x, edge_index, pred_missing, gen_feats):
        fill_feats, fill_edges = self.mend_graph(x, edge_index, pred_missing,
                                                 gen_feats)

        return fill_feats, fill_edges

class MendGraph_dgl(nn.Module):
    def __init__(self, num_pred):
        super(MendGraph_dgl, self).__init__()
        self.num_pred = num_pred
        for param in self.parameters():
            param.requires_grad = False

    def mend_graph(self, graph, pred_degree, gen_feats):
        device = gen_feats.device
        num_node, num_feature = graph.num_nodes(),graph.ndata['feat'].shape[1]
        new_edges = []
        gen_feats = gen_feats.view(-1, self.num_pred, num_feature)

        if pred_degree.device.type != 'cpu':
            pred_degree = pred_degree.cpu()
        pred_degree = torch._cast_Int(torch.round(pred_degree)).detach()
        x = graph.ndata['feat'].detach()
        fill_feats = torch.vstack((x, gen_feats.view(-1, num_feature)))
        mendg = dgl.add_nodes(graph, fill_feats.shape[0] - num_node)
        graph.ndata['feat'] = fill_feats
        for i in range(num_node):
            for j in range(min(self.num_pred, max(0, pred_degree[i]))):
                # dgl.add_edges(mendg, fill_edges[0], fill_edges[1])
                new_edges.append(
                    np.asarray([i, num_node + i * self.num_pred + j]))

        if len(new_edges) > 0:
            fill_edges = np.asarray(new_edges).reshape((-1, 2)).T
            fill_edges = torch.tensor(fill_edges, dtype=torch.int64).to(device)
            mendg = dgl.add_edges(mendg, fill_edges[0], fill_edges[1])
        return mendg

    def forward(self, graph, pred_missing, gen_feats):
        mendg = self.mend_graph(graph,  pred_missing, gen_feats)

        return mendg

class LocalSage_Plus(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden,
                 gen_hidden,
                 dropout=0.5,
                 num_pred=5):
        super(LocalSage_Plus, self).__init__()

        self.encoder_model = SAGE_Net(in_channels=in_channels,
                                      out_channels=gen_hidden,
                                      hidden=hidden,
                                      max_depth=2,
                                      dropout=dropout)
        self.reg_model = NumPredictor(latent_dim=gen_hidden)
        self.gen = FeatGenerator(latent_dim=gen_hidden,
                                 dropout=dropout,
                                 num_pred=num_pred,
                                 feat_shape=in_channels)
        self.mend_graph = MendGraph(num_pred)

        self.classifier = SAGE_Net(in_channels=in_channels,
                                   out_channels=out_channels,
                                   hidden=hidden,
                                   max_depth=2,
                                   dropout=dropout)

    def forward(self, data):

        x = self.encoder_model(data.x,data.edge_index)
        degree = self.reg_model(x)
        gen_feat = self.gen(x)
        mend_feats, mend_edge_index = self.mend_graph(data.x,data.edge_index,
                                                      degree, gen_feat)
        # nc_pred = self.classifier(
        #     Data(x=mend_feats, edge_index=mend_edge_index))
        nc_pred = self.classifier(mend_feats, mend_edge_index)
        return degree, gen_feat, nc_pred[:data.num_nodes]

    def inference(self, impared_data, raw_data):
        x = self.encoder_model(impared_data.x,impared_data.edge_index)
        degree = self.reg_model(x)
        gen_feat = self.gen(x)
        mend_feats, mend_edge_index = self.mend_graph(raw_data.x,
                                                      raw_data.edge_index,
                                                      degree, gen_feat)
        # nc_pred = self.classifier(
        #     Data(x=mend_feats, edge_index=mend_edge_index))
        nc_pred = self.classifier(mend_feats, mend_edge_index)
        return degree, gen_feat, nc_pred[:raw_data.num_nodes]

class LocalSage_Plus_gad(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden,
                 gen_hidden,
                 dropout=0.5,
                 num_pred=5,
                 negsamp_ratio=1,
                 readout='avg',
                 weight_decay=0.,
                 batch_size=0,
                 subgraph_size=4,
                 contamination=0.1,
                 verbose=False
                 ):
        super(LocalSage_Plus_gad, self).__init__()
        self.negsamp_ratio = negsamp_ratio
        self.readout = readout
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.subgraph_size = subgraph_size
        self.verbose = verbose
        self.encoder_model = SAGE_Net(in_channels=in_channels,
                                      out_channels=gen_hidden,
                                      hidden=hidden,
                                      max_depth=2,
                                      dropout=dropout)
        self.reg_model = NumPredictor(latent_dim=gen_hidden)
        self.gen = FeatGenerator(latent_dim=gen_hidden,
                                 dropout=dropout,
                                 num_pred=num_pred,
                                 feat_shape=in_channels)
        self.mend_graph = MendGraph(num_pred)

        self.gad = CoLA_Base(in_channels,
                               hidden,
                               'prelu',
                               self.negsamp_ratio,
                               self.readout,
                               self.subgraph_size
                               )
        # if self.batch_size:
        #     self.batch_num = self.num_nodes // self.batch_size + 1
        # else:  # full batch training
        #     self.batch_num = 1

        # self.classifier = SAGE_Net(in_channels=in_channels,
        #                            out_channels=out_channels,
        #                            hidden=hidden,
        #                            max_depth=2,
        #                            dropout=dropout)

    def forward(self, data):
        x = self.encoder_model(data.x,data.edge_index,)
        degree = self.reg_model(x)
        gen_feat = self.gen(x)
        mend_feats, mend_edge_index = self.mend_graph(data.x,data.edge_index,
                                                      degree, gen_feat)
        mend_x, mend_adj = process_graph( mend_feats, mend_edge_index)
        # graph anomany dection
        self.gad.train()
        all_idx = list(range(self.num_nodes))
        # random.shuffle(all_idx)
        subgraphs = self.generate_rw_subgraph(mend_edge_index, self.num_nodes,
                                         self.subgraph_size)

        cur_batch_size = len(all_idx)
        output = self.gad(mend_x,  mend_adj, all_idx, subgraphs, cur_batch_size)
        return degree, gen_feat, output[:data.num_nodes]

    def generate_rw_subgraph(self,edge_index, nb_nodes, subgraph_size):
        from torch_cluster import random_walk
        """Generate subgraph with random walk algorithm."""
        row, col = edge_index
        all_idx = torch.tensor(list(range(nb_nodes)))
        traces = random_walk(row, col, all_idx, walk_length=3)
        subv = traces.tolist()
        return subv




    def inference(self, impared_data, raw_data):
        x = self.encoder_model(impared_data)
        degree = self.reg_model(x)
        gen_feat = self.gen(x)
        mend_feats, mend_edge_index = self.mend_graph(raw_data.x,
                                                      raw_data.edge_index,
                                                      degree, gen_feat)
        nc_pred = self.classifier(
            Data(x=mend_feats, edge_index=mend_edge_index))
        return degree, gen_feat, nc_pred[:raw_data.num_nodes]

def process_graph(feat,edge_index):
    from torch_geometric.utils import to_dense_adj
    adj = to_dense_adj(edge_index)[0]
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    adj = (adj + sp.eye(adj.shape[0])).todense()

    x = torch.FloatTensor(feat[np.newaxis])
    adj = torch.FloatTensor(adj[np.newaxis])
    return x,adj
class LocalSage_Plus_dgl(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden,
                 gen_hidden,
                 dropout=0.5,
                 num_pred=5):
        super(LocalSage_Plus_dgl, self).__init__()
        self.encoder_model = GATv2Conv(in_feats =in_channels,
                                       out_feats = out_channels,
                                       num_heads=3)
        self.reg_model = NumPredictor(latent_dim=gen_hidden)
        self.gen = FeatGenerator(latent_dim=gen_hidden,
                                 dropout=dropout,
                                 num_pred=num_pred,
                                 feat_shape=in_channels)
        self.mend_graph = MendGraph_dgl(num_pred)

        self.gad_model = CoLA(in_feats =  in_channels)

        # self.classifier = SAGE_Net(in_channels=in_channels,
        #                            out_channels=out_channels,
        #                            hidden=hidden,
        #                            max_depth=2,
        #                            dropout=dropout)

    def forward(self, data):
        x = self.encoder_model(data)
        degree = self.reg_model(x)
        gen_feat = self.gen(x)
        mend_g = self.mend_graph(data, degree, gen_feat)
        self.gad_model.fit(mend_g, num_epoch=5, device=0)
        # nc_pred = self.classifier(
        #     Data(x=mend_feats, edge_index=mend_edge_index))
        # gad_result = self.gad_model.predict(g, auc_test_rounds=2, device=0)
        return degree, gen_feat #, nc_pred[:data.num_nodes]

    def inference(self, impared_data, raw_data):
        x = self.encoder_model(impared_data)
        degree = self.reg_model(x)
        gen_feat = self.gen(x)
        mend_g = self.mend_graph(raw_data, degree, gen_feat)
        self.gad_model.fit(mend_g, num_epoch=5, device=0)
        # nc_pred = self.classifier(
        #     Data(x=mend_feats, edge_index=mend_edge_index))
        return degree, gen_feat#, nc_pred[:raw_data.num_nodes]


class FedSage_Plus(nn.Module):
    def __init__(self, local_graph: LocalSage_Plus):
        super(FedSage_Plus, self).__init__()
        self.encoder_model = local_graph.encoder_model
        self.reg_model = local_graph.reg_model
        self.gen = local_graph.gen
        self.mend_graph = local_graph.mend_graph
        self.classifier = local_graph.classifier
        self.encoder_model.requires_grad_(False)
        self.reg_model.requires_grad_(False)
        self.mend_graph.requires_grad_(False)
        self.classifier.requires_grad_(False)

    def forward(self, data):
        x = self.encoder_model(data.x, data.edge_index,)
        degree = self.reg_model(x)
        gen_feat = self.gen(x)
        mend_feats, mend_edge_index = self.mend_graph(data.x, data.edge_index,
                                                      degree, gen_feat)
        # nc_pred = self.classifier(
        #     Data(x=mend_feats, edge_index=mend_edge_index))
        nc_pred = self.classifier(mend_feats,mend_edge_index)
        return degree, gen_feat, nc_pred[:data.num_nodes]



class FedSage_Plus_gad(nn.Module):
    def __init__(self, local_graph: LocalSage_Plus_gad):
        super(FedSage_Plus_gad, self).__init__()
        self.encoder_model = local_graph.encoder_model
        self.reg_model = local_graph.reg_model
        self.gen = local_graph.gen
        self.mend_graph = local_graph.mend_graph
        self.gad = local_graph.gad
        self.encoder_model.requires_grad_(False)
        self.reg_model.requires_grad_(False)
        self.mend_graph.requires_grad_(False)
        self.gad.requires_grad_(False)

    def forward(self, data):
        x = self.encoder_model(data)
        degree = self.reg_model(x)
        gen_feat = self.gen(x)
        mend_feats, mend_edge_index = self.mend_graph(data.x, data.edge_index,
                                                      degree, gen_feat)
        mend_x, mend_adj = process_graph(mend_feats, mend_edge_index)
        self.gad.train()
        all_idx = list(range(self.num_nodes))
        # random.shuffle(all_idx)
        subgraphs = self.generate_rw_subgraph(mend_edge_index, self.num_nodes,
                                              self.subgraph_size)

        cur_batch_size = len(all_idx)
        output = self.gad(mend_x, mend_adj, all_idx, subgraphs, cur_batch_size)

        return degree, gen_feat, output[:data.num_nodes]




class FedSage_Plus_dgl(nn.Module):
    def __init__(self, local_graph: LocalSage_Plus):
        super(FedSage_Plus_dgl, self).__init__()
        self.encoder_model = local_graph.encoder_model
        self.reg_model = local_graph.reg_model
        self.gen = local_graph.gen
        self.mend_graph = local_graph.mend_graph
        self.classifier = local_graph.classifier
        self.encoder_model.requires_grad_(False)
        self.reg_model.requires_grad_(False)
        self.mend_graph.requires_grad_(False)
        self.classifier.requires_grad_(False)

    def forward(self, graph):
        x = self.encoder_model(graph)
        degree = self.reg_model(x)
        gen_feat = self.gen(x)
        mend_g = self.mend_graph(graph,degree, gen_feat)
        self.gad_model.fit(mend_g, num_epoch=5, device=0)
        # nc_pred = self.classifier(
        #     Data(x=mend_feats, edge_index=mend_edge_index))
        return degree, gen_feat #, nc_pred[:data.num_nodes]
