from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Callable, List, Literal

import matplotlib.pyplot as plt
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from msr.training.data.datasets import MimicDataset, PtbXLDataset, SleepEDFDataset
from msr.training.data.utils import StratifiedBatchSampler
from msr.utils import align_left


class BaseDataModule(LightningDataModule, metaclass=ABCMeta):
    def __init__(
        self,
        representation_type: str,
        batch_size: int = 64,
        num_workers: int = 8,
        train_transform: Callable = None,
        inference_transform: Callable = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.representation_type = representation_type
        self.train_transform = train_transform
        self.inference_transform = inference_transform
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
            self.train = self.DatasetFactory(
                split="train", representation_type=self.representation_type, transform=self.train_transform
            )
            self.val = self.DatasetFactory(
                split="val", representation_type=self.representation_type, transform=self.inference_transform
            )
            self.datasets.extend([self.train, self.val])
        if stage == "test" or stage is None:
            self.test = self.DatasetFactory(
                split="test", representation_type=self.representation_type, transform=self.inference_transform
            )
            self.datasets.append(self.test)

    @property
    def input_shape(self):
        for ds in self.datasets:
            return ds.input_shape

    @property
    def transformed_input_shape(self):
        for ds in self.datasets:
            return ds.transformed_input_shape

    @property
    def num_classes(self):
        try:
            return len(self.class_names)
        except TypeError:
            return

    @property
    def class_names(self):
        for ds in self.datasets:
            try:
                return ds.class_names
            except AttributeError:
                return

    @property
    def feature_names(self):
        for ds in self.datasets:
            return ds.feature_names

    def get_transformed_data(self, split):
        split_dataset = getattr(self, split)
        split_data = split_dataset.data
        if split_dataset.transform is not None:
            return torch.stack([split_dataset.transform(sample) for sample in split_data])
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

    def get_X_y(self, split="all"):
        if split == "all":
            data = []
            for _split in ["train", "val", "test"]:
                data.extend(self.get_X_y(_split))
            return data
        else:
            X = getattr(self, f"{split}_data")
            y = getattr(self, split).targets
            return X, y

    def plot_targets(self, axes=None):
        ncols = len(self.datasets)
        return_fig = False
        if axes is None:
            return_fig = True
            fig, axes = plt.subplots(1, ncols, figsize=(ncols * 5, 5))
        for dataset, ax in zip(self.datasets, axes):
            dataset.plot_targets(ax=ax)
        axes[1].set_ylabel("")
        axes[2].set_ylabel("")
        plt.tight_layout()
        if return_fig:
            return fig

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_sampler=StratifiedBatchSampler(self.train.targets.int(), batch_size=self.batch_size, shuffle=True),
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_sampler=StratifiedBatchSampler(
                self.val.targets.int(), batch_size=10 * self.batch_size, shuffle=False
            ),
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_sampler=StratifiedBatchSampler(
                self.test.targets.int(), batch_size=10 * self.batch_size, shuffle=False
            ),
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
        train_transform: Callable = None,
        inference_transform: Callable = None,
    ):
        super().__init__(representation_type, batch_size, num_workers, train_transform, inference_transform)
        self.fs = fs
        self.target = target

    @property
    def DatasetFactory(self):
        return partial(PtbXLDataset, fs=self.fs, target=self.target)


class MimicDataModule(BaseDataModule):
    """Datamodule used for MIMIC dataset"""

    def __init__(
        self,
        representation_type: str,
        target: str = "sbp_dbp_avg",
        bp_targets: List[Literal["sbp", "dbp"]] = ["sbp"],
        batch_size: int = 64,
        num_workers=8,
        train_transform: Callable = None,
        inference_transform: Callable = None,
    ):
        super().__init__(representation_type, batch_size, num_workers, train_transform, inference_transform)
        self.target = target
        self.bp_targets = bp_targets

    @property
    def DatasetFactory(self):
        return partial(MimicDataset, target=self.target, bp_targets=self.bp_targets)


class SleepEDFDataModule(BaseDataModule):
    """Datamodule used for Sleep-EDF dataset"""

    @property
    def DatasetFactory(self):
        return SleepEDFDataset
