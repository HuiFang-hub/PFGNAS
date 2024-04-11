import torch

from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_networkx, from_networkx

import numpy as np
import networkx as nx

from src.federatedscope.core.splitters import BaseSplitter

EPSILON = 1e-5


class RandomSplitter(BaseTransform, BaseSplitter):
    """
    Split Data into small data via random sampling.

    Args:
        client_num (int): Split data into client_num of pieces.
        sampling_rate (str): Samples of the unique nodes for each client, \
            eg. ``'0.2,0.2,0.2'``
        overlapping_rate(float): Additional samples of overlapping data, \
            eg. ``'0.4'``
        drop_edge(float): Drop edges (drop_edge / client_num) for each \
            client within overlapping part.
    """
    def __init__(self,
                 client_num,
                 sampling_rate=None,
                 overlapping_rate=0,
                 drop_edge=0):
        BaseSplitter.__init__(self, client_num)
        self.ovlap = overlapping_rate
        if sampling_rate is not None:
            self.sampling_rate = np.array(
                [float(val) for val in sampling_rate.split(',')])
        else:
            # Default: Average
            self.sampling_rate = (np.ones(client_num) -
                                  self.ovlap) / client_num

        if len(self.sampling_rate) != client_num:
            raise ValueError(
                f'The client_num ({client_num}) should be equal to the '
                f'lenghth of sampling_rate and overlapping_rate.')

        if abs((sum(self.sampling_rate) + self.ovlap) - 1) > EPSILON:
            raise ValueError(
                f'The sum of sampling_rate:{self.sampling_rate} and '
                f'overlapping_rate({self.ovlap}) should be 1.')

        self.drop_edge = drop_edge

    def __call__(self, data, **kwargs):
        data.index_orig = torch.arange(data.num_nodes)
        data.train_mask, data.val_mask, data.test_mask = split_by_ratio(data.num_nodes)
        has_y_attribute = 'y' in data
        node_attrs = ['x', 'y', 'ay', 'train_mask', 'val_mask', 'test_mask'] if has_y_attribute else ['x', 'ay',
                                                                                                      'train_mask',
                                                                                                      'val_mask',
                                                                                                      'test_mask']

        G = to_networkx(
            data,
            node_attrs=['x', 'y', 'train_mask', 'val_mask', 'test_mask'],
            to_undirected=True)
        nx.set_node_attributes(G,
                               dict([(nid, nid)
                                     for nid in range(nx.number_of_nodes(G))]),
                               name="index_orig")

        client_node_idx = {idx: [] for idx in range(self.client_num)}

        indices = np.random.permutation(data.num_nodes)
        sum_rate = 0
        for idx, rate in enumerate(self.sampling_rate):
            client_node_idx[idx] = indices[round(sum_rate *
                                                 data.num_nodes):round(
                                                     (sum_rate + rate) *
                                                     data.num_nodes)]
            sum_rate += rate

        if self.ovlap:
            ovlap_nodes = indices[round(sum_rate * data.num_nodes):]
            for idx in client_node_idx:
                client_node_idx[idx] = np.concatenate(
                    (client_node_idx[idx], ovlap_nodes))

        # Drop_edge index for each client
        if self.drop_edge:
            ovlap_graph = nx.Graph(nx.subgraph(G, ovlap_nodes))
            ovlap_edge_ind = np.random.permutation(
                ovlap_graph.number_of_edges())
            drop_all = ovlap_edge_ind[:round(ovlap_graph.number_of_edges() *
                                             self.drop_edge)]
            drop_client = [
                drop_all[s:s + round(len(drop_all) / self.client_num)]
                for s in range(0, len(drop_all),
                               round(len(drop_all) / self.client_num))
            ]

        graphs = []
        for owner in client_node_idx:
            nodes = client_node_idx[owner]
            sub_g = nx.Graph(nx.subgraph(G, nodes))
            if self.drop_edge:
                sub_g.remove_edges_from(
                    np.array(ovlap_graph.edges)[drop_client[owner]])
            graphs.append(from_networkx(sub_g))

        return graphs

def split_by_ratio(num_data, frac_list=None, shuffle=False, random_state=None):
    if frac_list is None:
        frac_list = [0.8, 0.2, 0]
    frac_list = np.asarray(frac_list)
    assert np.allclose(np.sum(frac_list), 1.), \
        'Expect frac_list sum to 1, got {:.4f}'.format(np.sum(frac_list))
    lengths = (num_data * frac_list).astype(int)
    lengths[-1] = num_data - np.sum(lengths[:-1])

    if shuffle:
        indices = np.random.RandomState(
            seed=random_state).permutation(num_data)
    else:
        indices = np.arange(num_data)
    train_indices = indices[:lengths[0]]
    val_indices = indices[lengths[0]:lengths[0] + lengths[1]]
    test_indices = indices[lengths[0] + lengths[1]:]

    train_mask = torch.zeros(num_data, dtype=torch.bool)
    val_mask = torch.zeros(num_data, dtype=torch.bool)
    test_mask = torch.zeros(num_data, dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    return train_mask, val_mask, test_mask