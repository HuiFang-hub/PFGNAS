import os
import torch

import torch_geometric.datasets as datasets
import torch_geometric.transforms as T
#
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_scipy_sparse_matrix

from ogb.nodeproppred import PygNodePropPredDataset

def torch_save(base_dir, filename, data):

    file_path = os.path.join(base_dir, filename)
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory )
    torch.save(data,file_path)

def get_graph_data(config):
    data_path = config.data.root
    dataset = config.data.type.lower()
    if dataset.lower() in ['cora', 'citeseer', 'pubmed']:
        data = datasets.Planetoid(data_path, dataset, transform=T.Compose([LargestConnectedComponents(), T.NormalizeFeatures()]))[0]
    elif dataset.lower() in ['computers', 'photo']:
        data = datasets.Amazon(data_path, dataset, transform=T.Compose([LargestConnectedComponents(), T.NormalizeFeatures()]))[0]
        data.train_mask, data.val_mask, data.test_mask \
            = torch.zeros(data.num_nodes, dtype=torch.bool), torch.zeros(data.num_nodes, dtype=torch.bool), torch.zeros(data.num_nodes, dtype=torch.bool)
    elif dataset.lower() in ['ogbn-arxiv']:
        data = PygNodePropPredDataset(dataset, root=data_path, transform=T.Compose([T.ToUndirected(), LargestConnectedComponents()]))[0]
        data.train_mask, data.val_mask, data.test_mask \
            = torch.zeros(data.num_nodes, dtype=torch.bool), torch.zeros(data.num_nodes, dtype=torch.bool), torch.zeros(data.num_nodes, dtype=torch.bool)
        data.y = data.y.view(-1)
    num_classes = len(set(data.y.tolist()))
    # dataset = [ds for ds in dataset]
    # client_num = min(len(dataset), config.federate.client_num
    #                  ) if config.federate.client_num > 0 else len(dataset)
    # config.merge_from_list(['federate.client_num', client_num])
    config.merge_from_list(['model.num_classes', num_classes])
    config.merge_from_list(['model.input_shape', tuple(data.x.shape)])
    return data,config



def split_train(data, config,data_path):
    n_clients = config.federate.client_num

    ratio_train = config.data.ratio_train
    mode =  config.data.mode

    # data_path =config.data.root
    n_data = data.num_nodes
    ratio_test = (1-ratio_train)/2
    n_train = round(n_data * ratio_train)
    n_test = round(n_data * ratio_test)
    
    permuted_indices = torch.randperm(n_data)
    train_indices = permuted_indices[:n_train]
    test_indices = permuted_indices[n_train:n_train+n_test]
    val_indices = permuted_indices[n_train+n_test:]

    data.train_mask.fill_(False)
    data.test_mask.fill_(False)
    data.val_mask.fill_(False)

    data.train_mask[train_indices] = True
    data.test_mask[test_indices] = True
    data.val_mask[val_indices] = True

    torch_save(data_path, f'{mode}/{n_clients}/train.pt', {'data': data})
    torch_save(data_path, f'{mode}/{n_clients}/test.pt', {'data': data})
    torch_save(data_path, f'{mode}/{n_clients}/val.pt', {'data': data})
    print(f'splition done, n_train: {n_train}, n_test: {n_test}, n_val: {len(val_indices)}')
    return data

class LargestConnectedComponents(BaseTransform):
    r"""Selects the subgraph that corresponds to the
    largest connected components in the graph.

    Args:
        num_components (int, optional): Number of largest components to keep
            (default: :obj:`1`)
    """
    def __init__(self, num_components: int = 1):
        self.num_components = num_components

    def __call__(self, data: Data) -> Data:
        import numpy as np
        import scipy.sparse as sp

        adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)

        num_components, component = sp.csgraph.connected_components(adj)

        if num_components <= self.num_components:
            return data

        _, count = np.unique(component, return_counts=True)
        subset = np.in1d(component, count.argsort()[-self.num_components:])

        return data.subgraph(torch.from_numpy(subset).to(torch.bool))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.num_components})'
