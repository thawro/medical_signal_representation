from typing import List

import torch


class Flatten:
    def __init__(self, start_dim: int = 0, end_dim: int = -1):
        self.start_dim = start_dim
        self.end_dim = end_dim

    def __call__(self, sample: torch.Tensor):
        return sample.flatten(start_dim=self.start_dim, end_dim=self.end_dim)


class Permute:
    def __init__(self, dims: List[int]):
        self.dims = list(dims)

    def __call__(self, sample: torch.Tensor):
        return torch.permute(sample, self.dims)


class Downsample:
    def __init__(self, n: int, dim: int = 0):
        self.n = n
        self.dim = dim

    def __call__(self, sample: torch.Tensor):
        if self.dim == 0:
            return sample[:: self.n]
        elif self.dim == 1:
            return sample[:, :: self.n]
        elif self.dim == 2:
            return sample[:, :, :: self.n]
        else:
            return sample[..., :: self.n]
