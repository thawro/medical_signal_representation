from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Literal

import torch
from torch.utils.data import Dataset

from msr.data.representation.mimic import load_mimic_split
from msr.data.representation.ptbxl import load_ptbxl_split
from msr.data.representation.sleep_edf import load_sleep_edf_split


class BaseDataset(Dataset, ABC):
    """Base Dataset class used as representation data source for ML and DL models"""

    def __init__(self, split: Literal["train", "val", "test"], representation_type: str, transform: Callable = None):
        self.split = split
        self.representation_typee = representation_type
        self.transform = transform
        dataset = self.dataset_loader(split=split, representation_type=representation_type)
        self.data = dataset["data"]
        self.targets = dataset["targets"]
        self.info = dataset["info"]

    @property
    @abstractmethod
    def dataset_loader(self) -> Callable:
        """Must return Callable which when called returns dictionary with `data`, `targets` and `info` keys."""
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data[idx].float()
        data = torch.nan_to_num(data, nan=0.0)  # TODO !!!
        return data, self.targets[idx]


class PtbXLDataset(BaseDataset):
    """PTB-XL Dataset class used as representation data source for ML and DL models."""

    def __init__(self, split, representation_type, fs, target, transform=None):
        self.fs = fs
        self.target = target
        super().__init__(split, representation_type, transform)

    @property
    def dataset_loader(self) -> Callable:
        return partial(load_ptbxl_split, fs=self.fs, target=self.target)


class MimicDataset(BaseDataset):
    """Dataset class used for Mimic representations"""

    @property
    def dataset_loader(self) -> Callable:
        return load_mimic_split


class SleepEDFDataset(BaseDataset):
    """Dataset class used for Mimic representations"""

    @property
    def dataset_loader(self) -> Callable:
        return load_sleep_edf_split
