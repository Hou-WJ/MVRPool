from torch_geometric.transforms import BaseTransform
import torch


class MinMaxNormalize(BaseTransform):
    def __call__(self, data):
        x = data.x
        min_val = torch.min(x, dim=0)[0]
        max_val = torch.max(x, dim=0)[0]
        normalized_x = (x - min_val) / (max_val - min_val + 1e-8)
        data.x = normalized_x
        return data


class MeanNormalize(BaseTransform):
    def __call__(self, data):
        x = data.x
        mean = torch.mean(x, dim=0)
        std = torch.std(x, dim=0)
        normalized_x = (x - mean) / (std + 1e-8)
        data.x = normalized_x
        return data
