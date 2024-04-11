import numpy as np
import random
import torch
import torch.utils.data
import typing as _typing
from sklearn.model_selection import StratifiedKFold, KFold
from src.autogl import backend as _backend
from src.autogl.data import InMemoryDataset


def index_to_mask(index: torch.Tensor, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = True
    return mask


def random_splits_mask(
        dataset: InMemoryDataset,
        train_ratio: float = 0.2, val_ratio: float = 0.4,
        seed: _typing.Optional[int] = None
) -> InMemoryDataset:
    r"""If the data has masks for train/val/test, return the splits with specific ratio.

    Parameters
    ----------
    dataset : InMemoryDataset
        graph set
    train_ratio : float
        the portion of data that used for training.

    val_ratio : float
        the portion of data that used for validation.

    seed : int
        random seed for splitting dataset.
    """
    if not train_ratio + val_ratio <= 1:
        raise ValueError("the sum of provided train_ratio and val_ratio is larger than 1")

    def __random_split_masks(
            data,
            num_nodes: int
    ) -> _typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _rng_state: torch.Tensor = torch.get_rng_state()
        if seed is not None and isinstance(seed, int):
            torch.manual_seed(seed)
        perm = torch.randperm(num_nodes)
        train_index = perm[:int(num_nodes * train_ratio)]
        val_index = perm[int(num_nodes * train_ratio): int(num_nodes * (train_ratio + val_ratio))]
        test_index = perm[int(num_nodes * (train_ratio + val_ratio)):]
        torch.set_rng_state(_rng_state)
        data.train_mask = index_to_mask(train_index, num_nodes)
        data.val_mask = index_to_mask(val_index, num_nodes)
        data.test_mask = index_to_mask(test_index, num_nodes)
        return data

    if _backend.DependentBackend.is_pyg():
        dataset = [__random_split_masks(data, data.num_nodes) for data in dataset]
    else:
        pass
    return dataset

def random_splits_mask_class(
        dataset: InMemoryDataset,
        num_train_per_class: int = 20,
        num_val_per_class: int = 30,
        seed: _typing.Optional[int] = ...
):
    r"""If the data has masks for train/val/test, return the splits with specific number of samples from every class for training as suggested in Pitfalls of graph neural network evaluation [#]_ for semi-supervised learning.

    References
    ----------
    .. [#] Shchur, O., Mumme, M., Bojchevski, A., & Günnemann, S. (2018).
        Pitfalls of graph neural network evaluation.
        arXiv preprint arXiv:1811.05868.

    Parameters
    ----------
    dataset: InMemoryDataset
        instance of ``InMemoryDataset``
    num_train_per_class : int
        the number of samples from every class used for training.
    num_val_per_class : int
        the number of samples from every class used for validation.
    seed : int
        random seed for splitting dataset.
    """

    if _backend.DependentBackend.is_pyg():
        import torch_geometric.transforms as T
        transform = T.RandomNodeSplit(
            split='test_rest',
            num_train_per_class=num_train_per_class,
            num_val=num_val_per_class
        )
        dataset = [transform(data) for data in dataset]
    else:
        import dgl
        def __split_data(data):
            if (
                    'label' not in data.ndata
            ):
                return data
            elif 'label' in data.ndata:
                label: torch.Tensor = data.ndata['label']
            else:
                raise RuntimeError
            num_nodes: int = label.size(0)
            num_classes: int = label.cpu().max().item() + 1

            _rng_state: torch.Tensor = torch.get_rng_state()
            if seed not in (Ellipsis, None) and isinstance(seed, int):
                torch.manual_seed(seed)
            train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=label.device)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=label.device)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=label.device)
            for class_index in range(num_classes):
                idx = (label == class_index).nonzero().view(-1)
                assert num_train_per_class + num_val_per_class < idx.size(0), (
                    f"the total number of samples from every class "
                    f"used for training and validation is larger than "
                    f"the total samples in class [{class_index}]"
                )
                randomized_index: torch.Tensor = torch.randperm(idx.size(0))
                train_idx = idx[randomized_index[:num_train_per_class]]
                val_idx = idx[
                    randomized_index[num_train_per_class: (num_train_per_class + num_val_per_class)]
                ]
                train_mask[train_idx] = True
                val_mask[val_idx] = True
            else:
                remaining = (~(train_mask + val_mask)).nonzero().view(-1)
                test_mask[remaining] = True

            torch.set_rng_state(_rng_state)
            data.ndata["train_mask"] = train_mask
            data.ndata["val_mask"] = val_mask
            data.ndata["test_mask"] = test_mask
            if not torch.all(data.in_degrees() == 0):
                data = dgl.add_self_loop(data)
            return data
        new_dataset = []
        for graph_index in range(len(dataset)):
            new_dataset.append(__split_data(dataset[graph_index]))
        dataset = new_dataset
    return dataset

def graph_cross_validation(
        dataset: InMemoryDataset,
        n_splits: int = 10, shuffle: bool = True,
        random_seed: _typing.Optional[int] = ...,
        stratify: bool = False
) -> InMemoryDataset:
    r"""Cross validation for graph classification data

    Parameters
    ----------
    dataset : InMemoryDataset
        dataset with multiple graphs.

    n_splits : int
        the number of folds to split.

    shuffle : bool
        shuffle or not for sklearn.model_selection.StratifiedKFold

    random_seed : int
        random_state for sklearn.model_selection.StratifiedKFold

    stratify: bool
    """
    if not isinstance(n_splits, int):
        raise TypeError
    elif not n_splits > 0:
        raise ValueError
    if not isinstance(shuffle, bool):
        raise TypeError
    if not (random_seed in (Ellipsis, None) or isinstance(random_seed, int)):
        raise TypeError
    elif isinstance(random_seed, int) and random_seed >= 0:
        _random_seed: int = random_seed
    else:
        _random_seed: int = random.randrange(0, 65536)
    if not isinstance(stratify, bool):
        raise TypeError

    if stratify:
        kf = StratifiedKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=_random_seed
        )
    else:
        kf = KFold(
            n_splits=n_splits, shuffle=shuffle, random_state=_random_seed
        )
    if _backend.DependentBackend.is_pyg():
        dataset_y = np.array([x['y'].item() for x in dataset])
    else:
        dataset_y = np.array([x[1].item() for x in dataset])
    idx_list = [
        (train_index.tolist(), test_index.tolist())
        for train_index, test_index
        in kf.split(np.zeros(len(dataset)), np.array(dataset_y))
    ]

    # dataset.folds = idx_list
    setattr(dataset, 'folds', idx_list)
    setattr(dataset, 'train_index', idx_list[0][0])
    setattr(dataset, 'val_index', idx_list[0][1])
    return dataset


def set_fold(dataset: InMemoryDataset, fold_id: int) -> InMemoryDataset:
    r"""Set fold for graph dataset consist of multiple graphs.

    Parameters
    ----------
    dataset: `autogl.data.InMemoryDataset`
        dataset with multiple graphs.
    fold_id: `int`
        The fold in to use, MUST be in [0, dataset.n_splits)

    Returns
    -------
    `autogl.data.InMemoryDataset`
        The reference of original dataset.
    """
    if not (hasattr(dataset, 'folds') and dataset.folds is not None):
        raise ValueError("Dataset do NOT contain folds")
    if not 0 <= fold_id < len(dataset.folds):
        raise ValueError(
            f"Fold id {fold_id} exceed total cross validation split number {len(dataset.folds)}"
        )
    dataset.train_index = dataset.folds[fold_id][0]
    dataset.val_index = dataset.folds[fold_id][1]
    setattr(dataset, 'cross_train_split', [dataset[i] for i in dataset.train_index])
    setattr(dataset, 'cross_val_split', [dataset[i] for i in dataset.val_index])
    return dataset


def graph_random_splits(
        dataset: InMemoryDataset,
        train_ratio: float = 0.2,
        val_ratio: float = 0.4,
        seed: _typing.Optional[int] = ...
):
    r"""Splitting graph dataset with specific ratio for train/val/test.

    Parameters
    ----------
    dataset: ``InMemoryStaticGraphSet``

    train_ratio : float
        the portion of data that used for training.

    val_ratio : float
        the portion of data that used for validation.

    seed : int
        random seed for splitting dataset.
    """
    _rng_state = torch.get_rng_state()
    if isinstance(seed, int):
        torch.manual_seed(seed)
    perm = torch.randperm(len(dataset))
    train_index = perm[: int(len(dataset) * train_ratio)]
    val_index = (
        perm[int(len(dataset) * train_ratio): int(len(dataset) * (train_ratio + val_ratio))]
    )
    test_index = perm[int(len(dataset) * (train_ratio + val_ratio)):]
    train_index = train_index.tolist()
    val_index = val_index.tolist()
    test_index = test_index.tolist()
    dataset.train_index = train_index
    dataset.val_index = val_index
    dataset.test_index = test_index
    if not isinstance(dataset, InMemoryDataset):
        dataset.train_split = [dataset[i] for i in train_index]
        dataset.val_split = [dataset[i] for i in val_index]
        dataset.test_split = [dataset[i] for i in test_index]
    torch.set_rng_state(_rng_state)
    return dataset


def graph_get_split(
        dataset, mask: str = "train",
        is_loader: bool = True, batch_size: int = 128,
        num_workers: int = 0, shuffle: bool = False
) -> _typing.Union[torch.utils.data.DataLoader, _typing.Iterable]:
    r"""Get train/test dataset/dataloader after cross validation.

    Parameters
    ----------
    dataset:
        dataset with multiple graphs.

    mask : str

    is_loader : bool
        return original dataset or data loader

    batch_size : int
        batch_size for generating Dataloader
    num_workers : int
        number of workers parameter for data loader
    shuffle: bool
        whether shuffle the dataloader
    """
    if not isinstance(mask, str):
        raise TypeError
    elif mask.lower() not in ("train", "val", "test"):
        raise ValueError
    if not isinstance(is_loader, bool):
        raise TypeError
    if not isinstance(batch_size, int):
        raise TypeError
    elif not batch_size > 0:
        raise ValueError
    if not isinstance(num_workers, int):
        raise TypeError
    elif not num_workers >= 0:
        raise ValueError

    if mask.lower() not in ("train", "val", "test"):
        raise ValueError
    elif mask.lower() == "train":
        try:
            optional_dataset_split = dataset.train_split
        except AttributeError:
            optional_dataset_split = dataset.cross_train_split
        if optional_dataset_split is None:
            raise ValueError(f"Provided dataset do NOT have {mask} split")
        else:
            sub_dataset = InMemoryDataset(
                optional_dataset_split, train_index=list(range(len(optional_dataset_split)))
            )
    elif mask.lower() == "val":
        try:
            optional_dataset_split = dataset.val_split
        except AttributeError:
            optional_dataset_split = dataset.cross_val_split
        if optional_dataset_split is None:
            raise ValueError(f"Provided dataset do NOT have {mask} split")
        else:
            sub_dataset = InMemoryDataset(
                optional_dataset_split, val_index=list(range(len(optional_dataset_split)))
            )
    elif mask.lower() == "test":
        optional_dataset_split = dataset.test_split
        if optional_dataset_split is None:
            raise ValueError(f"Provided dataset do NOT have {mask} split")
        else:
            sub_dataset = InMemoryDataset(
                optional_dataset_split, test_index=list(range(len(optional_dataset_split)))
            )
    else:
        raise ValueError(
            f"The provided mask parameter must be a str in ['train', 'val', 'test'], "
            f"illegal provided value is [{mask}]"
        )
    if not is_loader:
        return sub_dataset
    if is_loader:
        if not (_backend.DependentBackend.is_dgl() or _backend.DependentBackend.is_pyg()):
            raise RuntimeError("Unsupported backend")
        elif _backend.DependentBackend.is_dgl():
            from dgl.dataloading import GraphDataLoader
            return GraphDataLoader(
                sub_dataset,
                **{"batch_size": batch_size, "num_workers": num_workers},
                shuffle=shuffle
            )
        elif _backend.DependentBackend.is_pyg():
            _sub_dataset: _typing.Any = optional_dataset_split
            import torch_geometric
            if int(torch_geometric.__version__.split('.')[0]) >= 2:
                # version 2.x
                from torch_geometric.loader import DataLoader
            else:
                from torch_geometric.data import DataLoader
            return DataLoader(
                _sub_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle
            )
    else:
        return sub_dataset
