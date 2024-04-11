import numpy as np
from src.federatedscope.core.splitters import BaseSplitter
from src.federatedscope.core.splitters.utils import \
    dirichlet_distribution_noniid_slice
import torch

class LDASplitter(BaseSplitter):
    """
    This splitter split dataset with LDA.

    Args:
        client_num: the dataset will be split into ``client_num`` pieces
        alpha (float): Partition hyperparameter in LDA, smaller alpha \
            generates more extreme heterogeneous scenario see \
            ``np.random.dirichlet``
    """
    def __init__(self, client_num, alpha=0.5):
        self.alpha = alpha
        super(LDASplitter, self).__init__(client_num)

    def __call__(self, dataset, prior=None, **kwargs):
        from torch.utils.data import Dataset, Subset
        from torch_geometric.data import Data
        label = dataset.y.numpy()
        index_orig_mapping = {nid: nid for nid in range(dataset.num_nodes)}

        # 在 Data 对象上添加 index_orig 属性
        dataset.index_orig = torch.tensor([index_orig_mapping[nid] for nid in range(dataset.num_nodes)])

        # tmp_dataset = [ds for ds in dataset]
        # label = np.array([y for x, y in tmp_dataset])
        idx_slice = dirichlet_distribution_noniid_slice(label,
                                                        self.client_num,
                                                        self.alpha,
                                                        prior=prior)
        if isinstance(dataset, Dataset):
            data_list = [Subset(dataset, idxs) for idxs in idx_slice]
        elif isinstance(dataset, Data):
            data_list = []
            for idx_list in idx_slice:
                # 从 dataset 中提取子图的节点和边信息
                subgraph_data = Data(
                    x=dataset.x[idx_list],
                    edge_index=dataset.edge_index[:, (dataset.edge_index[0].numpy() < len(idx_list)) & (
                                dataset.edge_index[1].numpy() < len(idx_list))],
                    y=dataset.y[idx_list],
                    train_mask=dataset.train_mask[idx_list],
                    val_mask=dataset.val_mask[idx_list],
                    test_mask=dataset.test_mask[idx_list],
                    index_orig=dataset.index_orig[idx_list]
                )
                data_list.append(subgraph_data)
        else:
            data_list = [[dataset[idx] for idx in idxs] for idxs in idx_slice]
        return data_list
