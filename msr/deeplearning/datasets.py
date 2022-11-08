from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Callable, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import Dataset

from msr.data.representation.mimic import load_mimic_split
from msr.data.representation.ptbxl import load_ptbxl_split
from msr.data.representation.sleep_edf import load_sleep_edf_split

sns.set(style="whitegrid")


class BaseDataset(Dataset, metaclass=ABCMeta):
    """Base Dataset class used as representation data source for ML and DL models"""

    def __init__(self, split: Literal["train", "val", "test"], representation_type: str, transform: Callable = None):
        super().__init__()
        self.split = split
        self.representation_type = representation_type
        self.transform = transform
        dataset = self._dataset_loader(split=split, representation_type=representation_type)
        self.data = dataset["data"]
        self.targets = dataset["targets"]
        self.info = dataset["info"]

    @property
    @abstractmethod
    def _dataset_loader(self) -> Callable:
        """Must return Callable which when called returns dictionary with `data`, `targets` and `info` keys."""
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data[idx].float()
        data = torch.nan_to_num(data, nan=0.0)  # TODO !!!
        if self.transform:
            data = self.transform(data)
        return data, self.targets[idx]

    @property
    def info_dict(self):
        return dict(
            representation_type=self.representation_type,
            split=self.split,
            data_shape=self.data.shape,
            targets=self.targets,
            info=self.info,
        )

    def __repr__(self):
        return f"{self.__class__ }\n" + "\n".join([f"   {name} = {value}" for name, value in self.info_dict.items()])


class ClassificationDataset(BaseDataset):
    def __init__(self, split: Literal["train", "val", "test"], representation_type: str, transform: Callable = None):
        super().__init__(split, representation_type, transform)
        self.classes = np.array([self.info[target] for target in self.targets])
        self.classes_counts = {target: count for target, count in zip(*np.unique(self.classes, return_counts=True))}
        self.classes_counts = dict(sorted(self.classes_counts.items(), key=lambda item: item[1], reverse=True))

    @property
    def info_dict(self):
        return {**super().info_dict, "classes_counts": self.classes_counts}

    def plot_classes_counts(self, ax=None):
        sns.countplot(x=self.classes, ax=ax, order=list(self.classes_counts.keys()))


class RegressionDataset(BaseDataset):
    def plot_classes_counts(self):
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.histplot(self.targets, ax=ax)


class PtbXLDataset(ClassificationDataset):
    """PTB-XL Dataset class used as representation data source for ML and DL models."""

    def __init__(
        self,
        split: Literal["train", "val", "test"],
        representation_type: str,
        fs: float,
        target: str,
        transform: Callable = None,
    ):
        self.fs = fs
        self.target = target
        super().__init__(split, representation_type, transform)

    @property
    def _dataset_loader(self) -> Callable:
        return partial(load_ptbxl_split, fs=self.fs, target=self.target)


class MimicDataset(RegressionDataset):
    """Dataset class used for Mimic representations"""

    @property
    def _dataset_loader(self) -> Callable:
        return load_mimic_split


class SleepEDFDataset(ClassificationDataset):
    """Dataset class used for Mimic representations"""

    @property
    def _dataset_loader(self) -> Callable:
        return load_sleep_edf_split
