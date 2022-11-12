from abc import abstractmethod
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from msr.evaluation.loggers import BaseWandbLogger
from msr.evaluation.metrics import ClafficationMetrics, RegressionMetrics
from msr.evaluation.plotters import BasePlotter, MatplotlibPlotter, PlotlyPlotter
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

    def train(self):
        return self.predict(self.datamodule.train.data.numpy())

    def validate(self):
        return self.predict(self.datamodule.val.data.numpy())

    def test(self):
        return self.predict(self.datamodule.test.data.numpy())

    @abstractmethod
    def plot_evaluation(
        self, y_values: Dict[str, Tuple[np.ndarray, np.ndarray]], metrics: Dict[str, float], plotter: BasePlotter
    ):
        pass

    def evaluate(self, plotter: BasePlotter = None, logger: BaseWandbLogger = None):
        all_y_values = {
            # "train": {"preds": self.train(), "target": self.datamodule.train.targets},
            "val": {"preds": self.validate(), "target": self.datamodule.val.targets},
            "test": {"preds": self.test(), "target": self.datamodule.test.targets},
        }

        metrics = {split: self.metrics.get_metrics(**y_values) for split, y_values in all_y_values.items()}
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
        self.metrics = ClafficationMetrics(num_classes=self.num_classes)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def train(self):
        return self.predict_proba(self.datamodule.train.data.numpy())

    def validate(self):
        return self.predict_proba(self.datamodule.val.data.numpy())

    def test(self):
        return self.predict_proba(self.datamodule.test.data.numpy())

    def plot_evaluation(
        self,
        y_values: Dict[str, Tuple[np.ndarray, np.ndarray]],
        metrics: Dict[str, float],
        plotter: BasePlotter = PlotlyPlotter(),
    ):
        filtered_metrics = {
            split: {metric: value for metric, value in split_metrics.items() if metric not in ["roc"]}
            for split, split_metrics in metrics.items()
        }
        figs = {
            "confusion_matrix": plotter.confusion_matrix(y_values, self.class_names),
            "roc": plotter.roc_curve(metrics, self.class_names),
            "metrics": plotter.metrics_comparison(filtered_metrics),
        }
        if hasattr(self.model, "feature_importances_"):
            figs["feature_importances"] = plotter.feature_importances(
                self.feature_names, self.model.feature_importances_, n_best=15
            )
        return figs


class MLRegressorTrainer(MLTrainer):
    def __init__(self, model: BasePredictor, datamodule: BaseDataModule):
        self.model = model
        self.datamodule = datamodule
        self.feature_names = datamodule.feature_names
        self.metrics = RegressionMetrics()

    def train(self):
        return self.predict(self.datamodule.train.data.numpy())

    def validate(self):
        return self.predict(self.datamodule.val.data.numpy())

    def test(self):
        return self.predict(self.datamodule.test.data.numpy())

    def plot_evaluation(
        self,
        y_values: Dict[str, Tuple[np.ndarray, np.ndarray]],
        metrics: Dict[str, float],
        plotter: BasePlotter = PlotlyPlotter(),
    ):
        figs = {"target_vs_preds": plotter.target_vs_preds(y_values), "metrics": plotter.metrics_comparison(metrics)}
        if hasattr(self.model, "feature_importances_"):
            figs["feature_importances"] = plotter.feature_importances(
                self.feature_names, self.model.feature_importances_, n_best=15
            )
        return figs
