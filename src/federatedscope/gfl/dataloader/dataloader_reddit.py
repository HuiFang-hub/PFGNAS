# -*- coding: utf-8 -*-
# @Time    : 2023/10/30 15:56
# @Function:
import torch
from torch_geometric.datasets import Reddit
from src.federatedscope.core.auxiliaries.splitter_builder import get_splitter
from src.federatedscope.core.auxiliaries.transform_builder import get_transform
import torch_geometric.transforms as T
from torch_geometric.utils import add_self_loops, remove_self_loops, \
    to_undirected


def load_nodelevel_reddit(config=None):
    path = config.data.root

    # 下载并加载 Reddit 数据集

    splitter = get_splitter(config)
    transforms_funcs, _, _ = get_transform(config, 'torch_geometric')
    dataset = Reddit(path,transform=T.NormalizeFeatures())
    global_data = copy.deepcopy(dataset)
    graph = splitter(dataset, seed=config.seed)
    dataset = splitter(graph)
    dataset = [ds for ds in dataset]
    client_num = min(len(dataset), config.federate.client_num
                     ) if config.federate.client_num > 0 else len(dataset)
    config.merge_from_list(['federate.client_num', client_num])
    data_dict = dict()
    num_nodes = []
    num_edges = []
    test_data = dataset[len(dataset) - 1]
    test_data.edge_index = add_self_loops(
        to_undirected(remove_self_loops(test_data.edge_index)[0]),
        num_nodes=test_data.x.shape[0])[0]
    for client_idx in range(1, len(dataset)):
        local_data = dataset[client_idx - 1]
        # To undirected and add self-loop
        local_data.edge_index = add_self_loops(
            to_undirected(remove_self_loops(local_data.edge_index)[0]),
            num_nodes=local_data.x.shape[0])[0]

        data_dict[client_idx] = {'data': local_data, 'test_data': test_data}
        num_nodes.append(local_data.num_nodes)
        num_edges.append(local_data.num_edges)

        # anomy_label.append(local_data.ay)
    # anomy_label = torch.cat(anomy_label)
    # global_data.ay =  anomy_label
    global_data.edge_index = add_self_loops(
        to_undirected(remove_self_loops(global_data.edge_index)[0]),
        num_nodes=global_data.x.shape[0])[0]
    data_dict[0] = {'data': global_data, 'test_data': test_data}
    return data_dict, config

