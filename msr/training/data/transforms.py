import torch


class Flatten:
    def __init__(self, start_dim: int = 0, end_dim: int = 1):
        self.start_dim = start_dim
        self.end_dim = end_dim

    def __call__(self, sample: torch.Tensor):
        return sample.flatten(start_dim=self.start_dim, end_dim=self.end_dim)
