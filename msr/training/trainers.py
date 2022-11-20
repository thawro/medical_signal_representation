from abc import abstractmethod
from functools import partial
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from msr.evaluation.loggers import MLWandbLogger
from msr.evaluation.metrics import get_classification_metrics, get_regression_metrics
from msr.evaluation.plotters import (
    BasePlotter,
    MatplotlibPlotter,
    PlotlyPlotter,
    plot_classifier_evaluation,
    plot_regressor_evaluation,
)
from msr.training.data.datamodules import BaseDataModule
from msr.training.utils import BasePredictor


class MLTrainer:
    def __init__(self, model: BasePredictor, datamodule: BaseDataModule):
        self.model = model
        self.datamodule = datamodule

    def fit(self):
        self.model.fit(X=self.datamodule.train.data.numpy(), y=self.datamodule.train.targets)

    def predict(self, X):
        return self.model.predict(X)

    @abstractmethod
    def plot_evaluation(
        self, y_values: Dict[str, Tuple[np.ndarray, np.ndarray]], metrics: Dict[str, float], plotter: BasePlotter
    ):
        pass

    def evaluate(self, plotter: BasePlotter = None, logger: MLWandbLogger = None):
        val_data = self.datamodule.val.data.numpy()
        val_targets = self.datamodule.val.targets

        test_data = self.datamodule.test.data.numpy()
        test_targets = self.datamodule.test.targets

        all_y_values = {
            # "train": {"preds": self.train(), "target": self.datamodule.train.targets},
            "val": {"preds": self.predict(val_data), "target": val_targets},
            "test": {"preds": self.predict(test_data), "target": test_targets},
        }

        metrics = {split: self.get_metrics(**y_values) for split, y_values in all_y_values.items()}
        evaluation_results = {
            "metrics": pd.json_normalize(metrics, sep="/").to_dict(orient="records")[0]  # flattened dict
        }
        if plotter is not None:
            evaluation_results["figs"] = self.plot_evaluation(all_y_values, metrics, plotter)
        if logger is not None:
            for name, results in evaluation_results.items():
                blacklist = ["/roc"]
                filtered_results = {
                    name: value for name, value in results.items() if all([key not in name for key in blacklist])
                }
                logger.log(filtered_results)
            logger.finish()
        return evaluation_results


class MLClassifierTrainer(MLTrainer):
    def __init__(self, model: BasePredictor, datamodule: BaseDataModule):
        self.model = model
        self.datamodule = datamodule
        self.class_names = datamodule.class_names
        self.feature_names = datamodule.feature_names
        self.num_classes = len(self.class_names)
        self.get_metrics = partial(get_classification_metrics, num_classes=self.num_classes)

    def predict(self, X):
        return self.model.predict_proba(X)

    def plot_evaluation(
        self,
        y_values: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
        metrics: Dict[str, Dict[str, float]],
        plotter: BasePlotter = PlotlyPlotter(),
    ):
        figs = plot_classifier_evaluation(
            y_values=y_values,
            metrics=metrics,
            class_names=self.class_names,
            feature_names=self.feature_names,
            feature_importances=self.model.feature_importances_,
            plotter=plotter,
        )
        return figs


class MLRegressorTrainer(MLTrainer):
    def __init__(self, model: BasePredictor, datamodule: BaseDataModule):
        self.model = model
        self.datamodule = datamodule
        self.feature_names = datamodule.feature_names
        self.get_metrics = get_regression_metrics

    def plot_evaluation(
        self,
        y_values: Dict[str, Tuple[np.ndarray, np.ndarray]],
        metrics: Dict[str, float],
        plotter: BasePlotter = PlotlyPlotter(),
    ):
        figs = plot_classifier_evaluation(
            y_values=y_values,
            metrics=metrics,
            feature_names=self.feature_names,
            feature_importances=self.model.feature_importances_,
            plotter=plotter,
        )
        return figs
