from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Callable, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import Dataset

from msr.data.create_representations.mimic import load_mimic_split
from msr.data.create_representations.ptbxl import load_ptbxl_split
from msr.data.create_representations.sleep_edf import load_sleep_edf_split
from msr.utils import align_left

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
        self.data = torch.nan_to_num(self.data)  # TODO
        self.targets = dataset["targets"]
        self.info = dataset["info"]
        if "feature_names" in dataset:
            self.feature_names = dataset["feature_names"]

    @property
    @abstractmethod
    def _dataset_loader(self) -> Callable:
        """Must return Callable which when called returns dictionary with `data`, `targets` and `info` keys."""
        pass

    @abstractmethod
    def plot_targets(self):
        """Plot targets distribution"""
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
            # representation_type=self.representation_type,
            data_shape=self.data.shape,
            targets=self.targets,
            info=self.info,
        )

    def describe(self, fields=["data_shape"]):
        info_dict_str = "\n".join([align_left(f"    {name}", self.info_dict[name], n_signs=25) for name in fields])
        return f"{self.split.title()} {self.__class__.__name__}:\n" + info_dict_str


class ClassificationDataset(BaseDataset):
    def __init__(self, split: Literal["train", "val", "test"], representation_type: str, transform: Callable = None):
        super().__init__(split, representation_type, transform)
        self.classes = np.array([self.info[target] for target in self.targets])
        self.classes_counts = {target: count for target, count in zip(*np.unique(self.classes, return_counts=True))}
        self.classes_counts = dict(sorted(self.classes_counts.items(), key=lambda item: item[1], reverse=True))
        self.class_names = sorted(self.info.values())
        self.classes_colors = {class_name: color for class_name, color in zip(self.class_names, sns.color_palette())}

    @property
    def info_dict(self):
        return {**super().info_dict, "classes_counts": self.classes_counts}

    def plot_targets(self, ax=None):
        order = list(self.classes_counts.keys())
        palette = [color[1] for color in sorted(self.classes_colors.items(), key=lambda pair: order.index(pair[0]))]
        sns.countplot(x=self.classes, ax=ax, order=order, palette=palette)


class RegressionDataset(BaseDataset):
    def plot_targets(self, ax=None):
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
