from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from deeplearning.datasets import PTBXLDataset
from deeplearning.utils import StratifiedBatchSampler


class PTBXLDataModule(LightningDataModule):
    """PTB-XL DataModule class used as DeepLearning models DataLoaders provider."""

    def __init__(
        self,
        representation_type: str,
        fs: float = 100,
        target: str = "diagnostic_class",
        batch_size: int = 64,
        num_workers=8,
    ):
        super().__init__()
        self.representation_type = representation_type
        self.fs = fs
        self.target = target
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train = None
        self.val = None
        self.test = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train = PTBXLDataset(
                self.representation_type,
                self.fs,
                self.target,
                split="train",
            )
            self.val = PTBXLDataset(self.representation_type, self.fs, self.target, split="val")
        if stage == "test" or stage is None:
            self.test = PTBXLDataset(self.representation_type, self.fs, self.target, split="test")

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
