import torch
import os.path as osp

from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.datasets import TUDataset, QM9, ZINC
from torch_geometric.transforms import Compose, NormalizeFeatures, OneHotDegree
from torch_geometric.utils import degree

from models.RewiredTUDataset import RewiredTUDataset
from normalization import MinMaxNormalize, MeanNormalize


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)

def max_node_nums(data):
    max_nums = 0
    for graph in data:
        if graph.num_nodes >= max_nums:
            max_nums = graph.num_nodes
    return max_nums

def build_viewlist_from_args(args):

    viewlist = []

    if args.original:
        viewlist.append({"type": "original"})

    if args.fully_connected:
        viewlist.append({"type": "fully_connected"})

    if args.order:
        viewlist.append({
            "type": "order",
            "order": args.order_num,
            "include_self_loops": args.include_self_loops,
            "normalize": args.order_norm,
            "threshold": args.order_threshold
        })

    if args.acyclic:
        viewlist.append({
            "type": "acyclic",
            "method": args.acyclic_method,
            "keep_important_edges": False,
            "importance_metric": "degree"
        })

    if args.spectral_embedding:
        viewlist.append({
            "type": "spectral_embedding",
            "k": args.spectral_embedding_k,
            "include_original": args.spectral_embedding_include_original
        })

    if args.curvature_based:
        viewlist.append({
            "type": "curvature_based",
            "curvature_type": args.curvature_type,
            "dynamic_update_fraction": args.dynamic_update_fraction,
            "use_long_range": args.use_long_range,
            "use_curvature_max": args.use_curvature_max
        })

    return viewlist


def load_dataset(dataset_name, path='./data', viewlist=None, data_normalization="None"):
    if dataset_name.startswith('ogbg-'):
        print(f"Loading OGB Dataset: {dataset_name}")

        dataset = PygGraphPropPredDataset(name=dataset_name, root=path)

        if dataset.task_type == 'subgenre classification' or 'classification' in dataset.task_type:
            task_type = 'classification'
        else:
            task_type = 'regression'

        if dataset[0].x is None and hasattr(dataset[0], 'num_nodes'):
            pass

        if viewlist is not None:
            print(
                "Warning: Rewired data for OGB is not fully implemented in RewiredTUDataset. Loading standard OGB dataset.")
        return dataset, task_type

    if dataset_name in ['QM9', 'ZINC_full']:
        task_type = 'regression'
    else:
        task_type = 'classification'

    if dataset_name in ["COLLAB", "IMDB-BINARY", "IMDB-MULTI", "QM9"]:
        dataset = TUDataset(path, dataset_name)
        max_degree = -1
        for data in dataset:
            d = degree(data.edge_index[0])
            max_degree = max(max_degree, int(d.max().item()))
        print(f"全局最大度: {max_degree}")
        data_normalization = OneHotDegree(max_degree=max_degree, cat=False)
    else:
        if data_normalization == "MaxMin":
            data_normalization = MinMaxNormalize()
        elif data_normalization == "Mean":
            data_normalization = MeanNormalize()
        else:
            data_normalization = None

    if viewlist is not None:
        dataset = RewiredTUDataset(root=path, name=dataset_name, viewlist=viewlist, transform=data_normalization)
        print("Load RewiredTUDataset")
    else:
        dataset = TUDataset(path, dataset_name, use_node_attr=True, transform=data_normalization)
        print("Load TUDataset")

    return dataset, task_type
