from abc import abstractmethod
from typing import Dict, List

import pytorch_lightning as pl
import torch
from torch import nn

from msr.evaluation.metrics import get_classification_metrics, get_regression_metrics


class BaseModule(pl.LightningModule):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self.save_hyperparameters(logger=False, ignore=["net"])

    @abstractmethod
    def get_metrics(self, preds, target, metrics):
        pass

    def forward(self, x):
        return self.net(x)

    def _common_step(self, batch, batch_idx: int, stage: str):
        data, target = batch
        preds = self.forward(data)
        loss = self.criterion(preds, target)
        return {"loss": loss, "preds": preds, "target": target}

    def training_step(self, batch, batch_idx: int):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx: int):
        return self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx: int):
        return self._common_step(batch, batch_idx, "test")

    def _common_epoch_end(self, outputs, stage: str):
        loss = torch.tensor([output["loss"] for output in outputs]).mean()
        preds = torch.cat([output["preds"] for output in outputs], dim=0)
        target = torch.cat([output["target"] for output in outputs], dim=0)
        metrics = self.get_metrics(preds, target)
        metrics["loss"] = loss.item()
        metrics = {f"{stage}/{name}": value for name, value in metrics.items()}
        results = {"metrics": metrics, "y_values": {"preds": preds, "target": target}}
        if self.trainer.sanity_checking or self.trainer.testing:
            return results
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.logger.log_metrics(metrics, step=self.current_epoch)
        return results

    def training_epoch_end(self, outputs: List):
        results = self._common_epoch_end(outputs, "train")
        trainer_metrics = {
            "epoch": self.current_epoch,
            "learning_rate": self.optimizers().param_groups[0]["lr"],
        }
        self.logger.log_metrics(trainer_metrics, step=self.current_epoch)

    def validation_epoch_end(self, outputs: List):
        results = self._common_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs: Dict):
        results = self._common_epoch_end(outputs, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=10,
            threshold=0.0001,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


class ClassifierModule(BaseModule):
    def __init__(self, net: nn.Module):
        super().__init__(net)
        self.criterion = nn.NLLLoss()

    def get_metrics(self, preds, target):
        metrics = ["accuracy", "fscore", "auroc"]
        return get_classification_metrics(num_classes=self.net.num_classes, preds=preds, target=target, metrics=metrics)


class RegressorModule(BaseModule):
    def __init__(self, net: nn.Module):
        super().__init__(net)
        self.criterion = nn.MSELoss()

    def get_metrics(self, preds, target):
        metrics = ["mae", "mape", "corr", "r2", "mse"]
        return get_regression_metrics(preds=preds, target=target, metrics=metrics)
