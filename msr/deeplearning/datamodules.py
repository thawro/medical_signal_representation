from abc import ABC, abstractmethod
from functools import partial

from deeplearning.datasets import MimicDataset, PtbXLDataset, SleepEDFDataset
from deeplearning.utils import StratifiedBatchSampler
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class BaseDataModule(LightningDataModule, ABC):
    def __init__(self, representation_type: str, batch_size: int = 64, num_workers: int = 8):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.representation_type = representation_type
        self.train = None
        self.val = None
        self.test = None

    @property
    @abstractmethod
    def DatasetFactory(self):
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train = self.DatasetFactory(split="train", representation_type=self.representation_type)
            self.val = self.DatasetFactory(split="val", representation_type=self.representation_type)
        if stage == "test" or stage is None:
            self.test = self.DatasetFactory(split="test", representation_type=self.representation_type)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_sampler=StratifiedBatchSampler(self.train[:, 1], batch_size=self.batch_size, shuffle=True),
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_sampler=StratifiedBatchSampler(self.val[:, 1], batch_size=10 * self.batch_size, shuffle=False),
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_sampler=StratifiedBatchSampler(self.test[:, 1], batch_size=10 * self.batch_size, shuffle=False),
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
    ):
        super().__init__(representation_type, batch_size, num_workers)
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
