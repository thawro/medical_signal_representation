import itertools
from abc import ABCMeta, abstractmethod
from typing import Callable, List, Literal, Tuple

import numpy as np
import seaborn as sns
import torch
from torch.utils.data import Dataset

from msr.data.dataset_providers import (
    MimicDatasetProvider,
    PtbXLDatasetProvider,
    SleepEDFDatasetProvider,
)
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
        self.data = dataset["data"].float()
        self.data = torch.nan_to_num(self.data)  # TODO
        self.targets = dataset["targets"]
        self.info = dataset["info"]
        if "feature_names" in dataset:
            self.feature_names = dataset["feature_names"]
            if len(self.feature_names.shape) > 1:
                self.feature_names = self.feature_names.flatten()
        else:
            shape_dims = [np.arange(_shape) for _shape in self.data.shape[1:]]
            feature_idxs = np.array(list(itertools.product(*shape_dims)))
            self.feature_names = np.array(["_".join(idxs.astype(str)) for idxs in feature_idxs])

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
        data = self.data[idx]
        data = torch.nan_to_num(data, nan=0.0)  # TODO !!!
        if self.transform:
            data = self.transform(data)
        return data.float(), self.targets[idx]

    @property
    def input_shape(self):
        return self.data[0].shape

    @property
    def transformed_input_shape(self):
        data, _ = self[0]
        return data.shape

    @property
    def info_dict(self):
        return dict(
            # representation_type=self.representation_type,
            n_samples=len(self),
            data_shape=self.data.shape,
            targets=self.targets,
            info=self.info,
        )

    def describe(self, fields=["input_shape"]):
        info_dict_str = "\n".join([align_left(f"    {name}", self.info_dict[name], n_signs=25) for name in fields])
        return f"{self.split.title()} {self.__class__.__name__}:\n" + info_dict_str


class ClassificationDataset(BaseDataset):
    def __init__(self, split: Literal["train", "val", "test"], representation_type: str, transform: Callable = None):
        super().__init__(split, representation_type, transform)
        self.classes = np.array([self.info[target.item()] for target in self.targets])
        self.classes_counts = {k: 0 for k in self.info.values()}
        self.classes_counts.update(
            {target: count for target, count in zip(*np.unique(self.classes, return_counts=True))}
        )
        self.classes_counts = dict(sorted(self.classes_counts.items(), key=lambda item: item[1], reverse=True))
        self.class_names = sorted([class_name for class_name, count in self.classes_counts.items() if count > 0])
        self.classes_colors = {class_name: color for class_name, color in zip(self.class_names, sns.color_palette())}

    @property
    def info_dict(self):
        return {**super().info_dict, "classes_counts": self.classes_counts}

    def plot_targets(self, ax=None):
        order = list(self.classes_counts.keys())
        palette = [color[1] for color in sorted(self.classes_colors.items(), key=lambda pair: order.index(pair[0]))]
        sns.countplot(x=self.classes, ax=ax, order=order, palette=palette)
        ax.tick_params(axis="x", labelrotation=75)


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
        target: Literal["diagnostic_class", "diagnostic_subclass"],
        transform: Callable = None,
    ):
        self.fs = fs
        self.target = target
        super().__init__(split, representation_type, transform)

    @property
    def _dataset_loader(self) -> Callable:
        ds_provider = PtbXLDatasetProvider(None, None, None, None, None, target=self.target)
        return ds_provider.load_split


class MimicDataset(RegressionDataset):
    """Dataset class used for Mimic representations"""

    def __init__(
        self,
        split: Literal["train", "val", "test"],
        representation_type: str,
        target: str,
        bp_targets: List[Literal["sbp", "dbp"]] = ["sbp"],
        transform: Callable = None,
    ):
        self.target = target
        self.bp_targets = bp_targets
        super().__init__(split, representation_type, transform)
        bp_targets_idxs = {"sbp": 0, "dbp": 1}
        bp_targets_idxs = [bp_targets_idxs[target] for target in bp_targets]
        if len(bp_targets) == 1:
            bp_targets_idxs = bp_targets_idxs[0]
        self.targets = self.targets[:, bp_targets_idxs].float()

    @property
    def _dataset_loader(self) -> Callable:
        ds_provider = MimicDatasetProvider(None, None, None, None, None, target=self.target)
        return ds_provider.load_split


class SleepEDFDataset(ClassificationDataset):
    """Dataset class used for Mimic representations"""

    @property
    def _dataset_loader(self) -> Callable:
        ds_provider = SleepEDFDatasetProvider(None, None)
        return ds_provider.load_split
