# -*- coding: utf-8 -*-
# @Time    : 30/03/2023 21:55
# @Function:
from torch_geometric.utils import add_self_loops, remove_self_loops, \
    to_undirected
import copy
import numpy as np
from torch_geometric.data import Data
from src.federatedscope.core.auxiliaries.utils import setup_seed
import src.federatedscope.register as register
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from src.federatedscope.core.splitters.graph import LouvainSplitter
from src.federatedscope.register import register_data
import torch
import os
import sys

import random
import community as community_louvain
import networkx as nx
import torch
from collections import Counter
from torch_geometric.datasets.attributed_graph_dataset import AttributedGraphDataset
from torch_geometric.datasets import Reddit2, Flickr, FacebookPagePage, Yelp, PolBlogs, Amazon, Twitch
import logging
import pickle
logger = logging.getLogger(__name__)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def my_cora(config=None):
    path = config.data.root

    num_split = [232, 542, np.iinfo(np.int64).max]
    dataset = Planetoid(path,
                        'cora',
                        split='random',
                        num_train_per_class=num_split[0],
                        num_val=num_split[1],
                        num_test=num_split[2])
    global_data = copy.deepcopy(dataset)[0]
    dataset = LouvainSplitter(config.federate.client_num)(dataset[0])

    data_local_dict = dict()
    for client_idx in range(len(dataset)):
        data_local_dict[client_idx + 1] = dataset[client_idx]

    data_local_dict[0] = global_data
    return data_local_dict, config


def splite_by_client_pyg(data, config, data_name):
    global_data = copy.deepcopy(data)
    # split setting
    client_num = config.federate.client_num
    if config.data.splitter_args:
        kwargs = config.data.splitter_args[0]
    else:
        kwargs = {}
    if config.data.splitter == 'louvain':
        from src.federatedscope.core.splitters.graph import LouvainSplitter_gad_pyg
        splitter = LouvainSplitter_gad_pyg(client_num, delta=20)
    dataset = splitter(data, data_name=data_name, seed=config.seed)
    # store in cilent dict
    dataset = [ds for ds in dataset]
    client_num = min(len(dataset), config.federate.client_num
                     ) if config.federate.client_num > 0 else len(dataset)
    config.merge_from_list(['federate.client_num', client_num])
    data_dict = dict()
    anomy_label = []
    for client_idx in range(1, len(dataset) + 1):
        local_data = dataset[client_idx - 1]
        # To undirected and add self-loop
        local_data.edge_index = add_self_loops(
            to_undirected(remove_self_loops(local_data.edge_index)[0]),
            num_nodes=local_data.x.shape[0])[0]
        data_dict[client_idx] = {'data': local_data}
        anomy_label.append(local_data.ay)
    anomy_label = torch.cat(anomy_label)
    global_data.ay = anomy_label
    data_dict[0] = {'data': global_data}
    return data_dict, config


def splite_by_client_pyg_before(data, config, data_name):
    global_data = copy.deepcopy(data)
    # split setting
    client_num = config.federate.client_num
    if config.data.splitter_args:
        kwargs = config.data.splitter_args[0]
    else:
        kwargs = {}
    if config.data.splitter == 'louvain':
        from src.federatedscope.core.splitters.graph import LouvainSplitter
        splitter = LouvainSplitter(client_num + 1, delta=config.data.splitter_delta)
        dataset = splitter(data, data_name=data_name, seed=config.seed)
    elif config.data.splitter == 'ScaffoldLdaSplitter':  # protein
        from src.federatedscope.core.splitters.graph import ScaffoldLdaSplitter
        splitter = ScaffoldLdaSplitter(client_num, alpha=config.data.splitter_args[0]['alpha'])
        dataset = splitter(data)
    elif config.data.splitter == 'RelTypeSplitter':
        from src.federatedscope.core.splitters.graph import RelTypeSplitter
        splitter = RelTypeSplitter(client_num, alpha=config.data.splitter_args[0]['alpha'])
        dataset = splitter(data)
    elif config.data.splitter == 'RandomSplitter':
        from src.federatedscope.core.splitters.graph import RandomSplitter
        splitter = RandomSplitter(client_num)
        dataset = splitter(data)
    elif config.data.splitter == 'random_partition':
        from src.federatedscope.core.splitters.graph import random_partition
        dataset, violin_df = random_partition(data, client_num + 1)

    # plot
    # plot_violinplot(violin_df)

    # store in cilent dict
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
        data_dict[client_idx] = {
            'data': local_data,
            'train': [local_data],
            'val': [local_data],
            'test': [local_data]
        }

        # anomy_label.append(local_data.ay)
    # anomy_label = torch.cat(anomy_label)
    # global_data.ay =  anomy_label
    global_data.edge_index = add_self_loops(
        to_undirected(remove_self_loops(global_data.edge_index)[0]),
        num_nodes=global_data.x.shape[0])[0]

    train_mask = torch.zeros_like(global_data.train_mask)
    val_mask = torch.zeros_like(global_data.val_mask)
    test_mask = torch.zeros_like(global_data.test_mask)

    for client_sampler in data_dict.values():
        if isinstance(client_sampler, Data):
            client_subgraph = client_sampler
        else:
            client_subgraph = client_sampler['data']
        train_mask[client_subgraph.index_orig[
            client_subgraph.train_mask]] = True
        val_mask[client_subgraph.index_orig[
            client_subgraph.val_mask]] = True
        test_mask[client_subgraph.index_orig[
            client_subgraph.test_mask]] = True
    global_data.train_mask = train_mask
    global_data.val_mask = val_mask
    global_data.test_mask = test_mask

    data_dict[0] = {
        'data': global_data,
        'train': [global_data],
        'val': [global_data],
        'test': [global_data]
    }
    
    # data_dict[0] = {'data': global_data, 'test_data': test_data}
    return data_dict, config, num_nodes, num_edges


def call_my_data(config, client_cfgs=None):
    import os.path as osp
    # basepath= osp.dirname(__file__)
    path = config.data.root
    data_name = config.data.type.lower()
    seed = random.randint(1, 100)
    graph = None
    modified_config = None, None
    if data_name == "lls":
        # filepath = path+'lls/feature_processed.data'
        filepath = './data/lls/feature_processed_one_hot_label.data'
        with open(filepath, 'rb') as f:
            graph = pickle.load(f)
    graph.x = graph.x.to(torch.float)
    # graph.y = graph.y.to(torch.float)
    #add mask
    # 创建节点掩码
    num_nodes =  graph.y.shape[0]
    num_train = int(0.7 * num_nodes)
    num_val = int(0.2 * num_nodes)
    num_test = num_nodes - num_train - num_val
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    # 设置掩码值
    train_mask[:num_train] = True
    val_mask[num_train:num_train + num_val] = True
    test_mask[num_train + num_val:] = True
    # 构建 PyG 的 Data 对象并添加掩码
    graph.train_mask=train_mask
    graph.val_mask=val_mask
    graph.test_mask=test_mask

    # logger.info(f'y proportion:{Counter(graph.y.tolist())}')
    # num_classes = len(set(graph.y.tolist()))
    config.model.num_classes = len(set(graph.y.tolist())) # graph.y.shape[1]
    config.model.input_shape= tuple(graph.x.shape)#len(torch.unique(graph.y))

    sum_nodes = graph.num_nodes
    sum_edges = graph.num_edges
    data, modified_config, num_nodes, num_edges = splite_by_client_pyg_before(graph, config, data_name)
    logger.info(f'num_nodes:{num_nodes} sum_nodes:{sum_nodes} average_num_nodes:{np.mean(num_nodes)}')
    logger.info(f'num_edges:{num_edges} sum_edges:{sum_edges} average_num_edges:{np.mean(num_edges)}')

    return data, modified_config




register_data("lls", call_my_data)

# -*- coding: utf-8 -*-
# @Time    : 2024/1/26 10:20
# @Function:
