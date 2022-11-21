from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Callable

import matplotlib.pyplot as plt
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from msr.training.data.datasets import MimicDataset, PtbXLDataset, SleepEDFDataset
from msr.training.data.utils import StratifiedBatchSampler
from msr.utils import align_left


class BaseDataModule(LightningDataModule, metaclass=ABCMeta):
    def __init__(
        self, representation_type: str, batch_size: int = 64, num_workers: int = 8, transform: Callable = None
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.representation_type = representation_type
        self.transform = transform
        self.dataset_params = dict(representation_type=representation_type, transform=transform)
        self.datasets = []

    def describe(self, ds_fields=["data_shape", "classes_counts"], dm_fields=["info"]):
        dm_connector = "\n" if dm_fields else ""
        ds_connector = "\n\n  " if ds_fields else ""
        datamodule_str = dm_connector.join(
            [align_left(f"{name}", self.datasets[0].info_dict[name], n_signs=10) for name in dm_fields]
        )
        datasets_str = ds_connector.join([dataset.describe(ds_fields) for dataset in self.datasets])
        return (
            f"{self.__class__.__name__} ({self.representation_type})\n  "
            + f"{datamodule_str}\n{dm_connector}  "
            + datasets_str
        )

    @property
    @abstractmethod
    def DatasetFactory(self):
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train = self.DatasetFactory(split="train", **self.dataset_params)
            self.val = self.DatasetFactory(split="val", **self.dataset_params)
            self.datasets.extend([self.train, self.val])
        if stage == "test" or stage is None:
            self.test = self.DatasetFactory(split="test", **self.dataset_params)
            self.datasets.append(self.test)
        for dataset in self.datasets:
            if hasattr(dataset, "feature_names"):
                self.feature_names = dataset.feature_names
                self.class_names = dataset.class_names
                self.num_classes = len(self.class_names)
                break

    def get_transformed_data(self, split):
        split_data = getattr(self, split).data
        if self.transform is not None:
            return torch.stack([self.transform(sample) for sample in split_data])
        else:
            return split_data

    @property
    def train_data(self):
        return self.get_transformed_data("train")

    @property
    def val_data(self):
        return self.get_transformed_data("val")

    @property
    def test_data(self):
        return self.get_transformed_data("test")

    def plot_targets(self):
        ncols = len(self.datasets)
        fig, axes = plt.subplots(1, ncols, figsize=(ncols * 5, 3.5))
        for dataset, ax in zip(self.datasets, axes):
            dataset.plot_targets(ax=ax)
            ax.set_title(dataset.split)
        plt.tight_layout()

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_sampler=StratifiedBatchSampler(self.train[:][1], batch_size=self.batch_size, shuffle=True),
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_sampler=StratifiedBatchSampler(self.val[:][1], batch_size=10 * self.batch_size, shuffle=False),
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_sampler=StratifiedBatchSampler(self.test[:][1], batch_size=10 * self.batch_size, shuffle=False),
            num_workers=self.num_workers,
        )


class PtbXLDataModule(BaseDataModule):
    """Datamodule used for PTB-XL dataset"""

    def __init__(
        self,
        representation_type: str,
        fs: float = 100,
        target: str = "diagnostic_class",
        batch_size: int = 64,
        num_workers=8,
        transform: Callable = None,
    ):
        super().__init__(representation_type, batch_size, num_workers, transform)
        self.fs = fs
        self.target = target

    @property
    def DatasetFactory(self):
        return partial(PtbXLDataset, fs=self.fs, target=self.target)


class MimicDataModule(BaseDataModule):
    """Datamodule used for MIMIC dataset"""

    @property
    def DatasetFactory(self):
        return MimicDataset


class SleepEDFDataModule(BaseDataModule):
    """Datamodule used for Sleep-EDF dataset"""

    @property
    def DatasetFactory(self):
        return SleepEDFDataset
