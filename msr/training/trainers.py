from abc import abstractmethod
from functools import partial
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch import nn

from msr.evaluation.metrics import get_classification_metrics, get_regression_metrics
from msr.evaluation.plotters import (
    BasePlotter,
    plot_classifier_evaluation,
    plot_regressor_evaluation,
)
from msr.training.data.datamodules import BaseDataModule
from msr.training.loggers import MLWandbLogger
from msr.training.utils import BasePredictor


class BaseTask:
    @abstractmethod
    def get_metrics(self, preds, target, metrics):
        pass

    @abstractmethod
    def plot_evaluation(
        self,
        y_values: Dict[str, Tuple[np.ndarray, np.ndarray]],
        metrics: Dict[str, float],
        plotter: BasePlotter,
        feature_importances: List[float] = None,
    ):
        pass


class Classifier:
    def get_metrics(self, preds, target):
        metrics = ["accuracy", "fscore", "auroc", "auc", "roc"]
        return get_classification_metrics(
            num_classes=self.datamodule.num_classes, preds=preds, target=target, metrics=metrics
        )

    def plot_evaluation(
        self,
        y_values: Dict[str, Tuple[np.ndarray, np.ndarray]],
        metrics: Dict[str, float],
        plotter: BasePlotter,
        feature_importances: List[float] = None,
    ):
        return plot_classifier_evaluation(
            y_values=y_values,
            metrics=metrics,
            class_names=self.datamodule.class_names,
            feature_names=self.feature_names,
            feature_importances=feature_importances,
            plotter=plotter,
        )


class Regressor:
    def get_metrics(self, preds, target):
        metrics = ["mae", "mape", "corr", "r2", "mse"]
        return get_regression_metrics(preds=preds, target=target, metrics=metrics)

    def plot_evaluation(
        self,
        y_values: Dict[str, Tuple[np.ndarray, np.ndarray]],
        metrics: Dict[str, float],
        plotter: BasePlotter,
        feature_importances: List[float] = None,
    ):
        return plot_regressor_evaluation(
            y_values=y_values,
            metrics=metrics,
            feature_names=self.feature_names,
            feature_importances=feature_importances,
            plotter=plotter,
        )


class BaseTrainer:
    def __init__(self, model, datamodule):
        self.model = model
        self.datamodule = datamodule
        self.feature_names = datamodule.feature_names

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self, data):
        pass

    @abstractmethod
    def evaluate(self, plotter: BasePlotter = None, logger: MLWandbLogger = None):
        all_y_values = {
            # "train": {"preds": self.predict(self.datamodule.train.data), "target": self.datamodule.train.targets},
            "val": {"preds": self.predict(self.datamodule.val_data), "target": self.datamodule.val.targets},
            "test": {"preds": self.predict(self.datamodule.test_data), "target": self.datamodule.test.targets},
        }

        metrics = {split: self.get_metrics(**y_values) for split, y_values in all_y_values.items()}
        evaluation_results = {
            "metrics": pd.json_normalize(metrics, sep="/").to_dict(orient="records")[0]  # flattened dict
        }
        if plotter is not None:
            evaluation_results["figs"] = self.plot_evaluation(
                y_values=all_y_values,
                metrics=metrics,
                plotter=plotter,
                feature_importances=getattr(self.model, "feature_importances_", None),
            )
        if logger is not None:
            for name, results in evaluation_results.items():
                blacklist = ["/roc"]
                filtered_results = {
                    name: value for name, value in results.items() if all([key not in name for key in blacklist])
                }
                logger.log(filtered_results)
            logger.finish()
        return evaluation_results


class DLTrainer(BaseTrainer):
    def __init__(self, trainer: pl.Trainer, model: nn.Module, datamodule: pl.LightningDataModule):
        self.trainer = trainer
        super().__init__(model, datamodule)

    def fit(self):
        self.trainer.fit(self.model, self.datamodule)

    def predict(self, data):
        return self.model(data)


class DLClassifierTrainer(DLTrainer, Classifier):
    def __init__(self, trainer: pl.Trainer, model: nn.Module, datamodule: pl.LightningDataModule):
        super().__init__(trainer, model, datamodule)


class DLRegressorTrainer(DLTrainer, Regressor):
    def __init__(self, trainer: pl.Trainer, model: nn.Module, datamodule: pl.LightningDataModule):
        super().__init__(trainer, model, datamodule)


class MLTrainer(BaseTrainer):
    def __init__(self, model, datamodule, normalize=True):
        self.model = model
        self.datamodule = datamodule
        self.feature_names = datamodule.feature_names

    def fit(self):
        self.model.fit(X=self.datamodule.train_data.numpy(), y=self.datamodule.train.targets)

    def predict(self, X):
        return self.model.predict(X)


class MLClassifierTrainer(MLTrainer, Classifier):
    def __init__(self, model, datamodule: pl.LightningDataModule):
        super().__init__(model, datamodule)

    def predict(self, X):
        return self.model.predict_proba(X)


class MLRegressorTrainer(MLTrainer, Regressor):
    def __init__(self, model, datamodule: pl.LightningDataModule):
        super().__init__(model, datamodule)
