import torch

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_networkx, from_networkx

import networkx as nx
import numpy as np

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
        G = to_networkx(data,
                        node_attrs=[
                            'x', 'y', 'train_mask', 'val_mask', 'test_mask',
                            'index_orig', 'ids_missing'
                        ],
                        to_undirected=True)

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
            G.nodes[i]['num_missing'] = np.array([len(ids_missing)],
                                                 dtype=np.float32)
            if len(ids_missing) > 0:
                if len(ids_missing) <= self.num_pred:
                    # test = data.x[ids_missing]
                    # test1 = np.zeros((self.num_pred - len(ids_missing),
                    #                data.x.shape[1]))
                    G.nodes[i]['x_missing'] = np.vstack(
                        (data.x[ids_missing],
                         np.zeros((self.num_pred - len(ids_missing),
                                   data.x.shape[1]))))
                    # test3 = G.nodes[i]['x_missing']
                else:
                    G.nodes[i]['x_missing'] = data.x[
                        ids_missing[:self.num_pred]]
            else:
                G.nodes[i]['x_missing'] = np.zeros(
                    (self.num_pred, data.x.shape[1]))

        # 将处理过的NetworkX图对象转换为DGL图对象并返回
        return from_networkx(nx.subgraph(G, remaining_nodes))

    def __repr__(self):
        return f'{self.__class__.__name__}({self.hidden_portion})'


class HideGraph_dgl(BaseTransform):
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
        val_ids = torch.where(data.ndata['val_mask'] == True)[0]
        # 从验证集中随机选择一定比例的节点进行隐藏
        hide_ids = np.random.choice(val_ids,
                                    int(len(val_ids) * self.hidden_portion),
                                    replace=False)
        # 新建一个掩码，用来表示哪些节点被保留下来了
        remaining_mask = torch.ones(data.num_nodes(), dtype=torch.bool)
        # 被隐藏的节点的掩码变成False
        remaining_mask[hide_ids] = False
        # 找到保留下来的节点编号
        remaining_nodes = torch.where(remaining_mask == True)[0].numpy()
        # 为每个节点添加一个列表，用来存储那些在该节点的邻居中被隐藏的节点的编号
        data.ndata['ids_missing'] = torch.tensor([[] for _ in range(data.num_nodes())])
        # 将DGL图对象转换为NetworkX图对象，便于操作
        G = dgl.to_networkx(
            data,
            node_attrs=['feat', 'label', 'train_mask', 'val_mask', 'test_mask', 'index_orig','ids_missing']).to_undirected()

        # 遍历所有被隐藏的节点的邻居，并在其邻居节点的节点信息中的ids_missing列表中添加该被隐藏的节点的编号
        for missing_node in hide_ids:
            neighbors = G.neighbors(missing_node)
            for i in neighbors:
                G.nodes[i]['ids_missing'] = torch.cat([ G.nodes[i]['ids_missing'], torch.tensor([missing_node])],dim=0)
                # G.nodes[i]['ids_missing'].append(missing_node)
        # 遍历所有节点，处理被隐藏的节点的信息，将它们的ids_missing列表删除，用num_missing字段记录被隐藏节点的数量
        # 用x_missing字段记录被隐藏的节点的特征向量（如果它们的邻居中被隐藏节点的数量小于等于num_pred，则特征向量的长度用0补齐，
        # 否则仅取前num_pred个被隐藏的节点的特征向量）

        for i in G.nodes:
            ids_missing = G.nodes[i]['ids_missing'].long().tolist()
            del G.nodes[i]['ids_missing']
            G.nodes[i]['num_missing'] = torch.tensor([len(ids_missing)], dtype=torch.float32)
            if len(ids_missing) > 0:
                if len(ids_missing) <= self.num_pred:
                    G.nodes[i]['x_missing'] = torch.vstack((
                        data.ndata['feat'][ids_missing],
                        torch.zeros((self.num_pred - len(ids_missing),data.ndata['feat'].shape[1]))
                    ))

                else:
                    G.nodes[i]['x_missing'] = data.ndata['feat'][
                        ids_missing[:self.num_pred]]
            else:
                G.nodes[i]['x_missing'] = torch.zeros(
                    (self.num_pred, data.ndata['feat'].shape[1]))

        # 将处理过的NetworkX图对象转换为DGL图对象并返回
        return dgl.from_networkx(nx.subgraph(G, remaining_nodes), node_attrs=['feat', 'label','train_mask','val_mask','test_mask','index_orig','x_missing'])

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
    filled_data = Data(
        x=new_features,
        edge_index=new_edge_index.T,
        train_idx=torch.where(original_data.train_mask == True)[0],
        valid_idx=torch.where(original_data.val_mask == True)[0],
        test_idx=torch.where(original_data.test_mask == True)[0],
        y=new_y,
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
