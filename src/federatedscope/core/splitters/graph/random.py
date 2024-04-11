# -*- coding: utf-8 -*-
# @Time    : 05/06/2023 15:51
# @Function:
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges
from torch_geometric.utils import subgraph
# def random_partition(data, n):
#     num_nodes = len(data.x)  # 获取大图数据集中的节点数
#     # node_indices = torch.arange(num_nodes)  # 创建包含所有节点索引的张量
#     subgraphs = []
#     for _ in range(n):
#         # 随机选择一部分节点作为子图的节点
#         subgraph_node_indices = torch.randperm(num_nodes)[:num_nodes // n]
#         # 使用subgraph函数生成子图数据对象
#         src, dst = subgraph(subgraph_node_indices, data.edge_index)[0]
#         subgraph_data = Data(
#             x=data.x[subgraph_node_indices],
#             edge_index=torch.stack([src, dst], dim=0),
#             ay=data.ay[subgraph_node_indices]
#         )
#         subgraphs.append(subgraph_data)
#
#     return subgraphs

def random_partition(data, num_clients):
    num_nodes = data.num_nodes
    num_edges = data.num_edges
    batch_size = int(num_nodes / num_clients) + 1


    violin_dict = {'client_id':[],'y_label':[],'anomaly_label':[]}
    # violin_df = pd.DataFrame(violin_keys)
    splitted_graphs = []
    for i in range(num_clients):
        start_idx = i * batch_size
        if i == num_clients-1:
            end_idx = num_nodes
        else:
            end_idx = (i + 1) * batch_size

        mask = (data.edge_index[0] >= start_idx) & (data.edge_index[0] < end_idx)&(data.edge_index[1] >= start_idx) & (data.edge_index[1] < end_idx)
        edge_index = data.edge_index[:, mask]
        # set
        # test3 = torch.unique( edge_index.view(-1))
        edge_index[0] -= start_idx
        edge_index[1] -= start_idx

        node_attr = data.x[start_idx:end_idx] if data.x is not None else None
        # edge_attr = data.edge_attr[mask] if data.edge_attr is not None else None
        ay = data.ay[start_idx:end_idx] if data.ay is not None else None
        y =  data.y[start_idx:end_idx] if data.y is not None else None
        test = [i for _ in range(end_idx - start_idx)]

        # violin_df = pd.concat([violin_df, pd.DataFrame({'x_id':[i for _ in range(end_idx - start_idx)]})], ignore_index=True)
        # violin_df = pd.concat([violin_df, pd.DataFrame({'y_label':y.tolist()})],ignore_index=True)
        # violin_df = pd.concat([violin_df, pd.DataFrame({'hub_label': ay.tolist()})], ignore_index=True)
        violin_dict['client_id'] += [i for _ in range(end_idx - start_idx)]
        violin_dict['y_label'] += y.tolist()
        violin_dict['anomaly_label'] += ay.tolist()

        subgraph = Data(x=node_attr, y=y,edge_index=edge_index, ay=ay)
        # if 'val_mask' not in subgraph:
        subgraph.train_mask, subgraph.val_mask, subgraph.test_mask = split_by_ratio(subgraph.num_nodes)
        subgraph.index_orig = torch.arange(subgraph.num_nodes)
        splitted_graphs.append(subgraph)
    violin_df = pd.DataFrame(violin_dict)
    return splitted_graphs,violin_df

def split_by_ratio(num_data, frac_list=None, shuffle=False, random_state=None):
    if frac_list is None:
        frac_list = [0.7, 0.2, 0.1]
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
