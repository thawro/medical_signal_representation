from abc import abstractmethod
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from msr.evaluation.metrics import ClafficationMetrics, RegressionMetrics
from msr.evaluation.visualisations import BasePlotter, MatplotlibPlotter, PlotlyPlotter
from msr.training.data.datamodules import BaseDataModule
from msr.training.utils import BasePredictor


class MLTrainer:
    def __init__(self, model: BasePredictor, datamodule: BaseDataModule):
        self.model = model
        self.datamodule = datamodule

    def fit(self):
        self.model.fit(X=self.datamodule.train.data.numpy(), y=self.datamodule.train.targets)

    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def plot_evaluation(
        self, y_values: Dict[str, Tuple[np.ndarray, np.ndarray]], metrics: Dict[str, float], plotter: BasePlotter
    ):
        pass

    def evaluate(self, plotter: BasePlotter = None):
        y_pred_val = self.validate()
        y_pred_test = self.test()
        y_values = {
            "val": {"preds": y_pred_val, "target": self.datamodule.val.targets},
            "test": {"preds": y_pred_test, "target": self.datamodule.test.targets},
        }

        metrics = {
            "val": self.metrics.get_metrics(**y_values["val"]),
            "test": self.metrics.get_metrics(**y_values["test"]),
        }

        if plotter is not None:
            self.plot_evaluation(y_values, metrics, plotter)

        return metrics


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

    def predict(self, X):
        return self.model.predict(X)

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
        plotter.confusion_matrix(y_values, self.class_names)
        plotter.roc_curve(metrics, self.class_names)
        if hasattr(self.model, "feature_importances_"):
            plotter.feature_importances(self.feature_names, self.model.feature_importances_, n_best=15)


class MLRegressorTrainer(MLTrainer):
    def __init__(self, model: BasePredictor, datamodule: BaseDataModule):
        self.model = model
        self.datamodule = datamodule
        self.metrics = RegressionMetrics()

    def validate(self):
        return self.model.predict(self.datamodule.val.data.numpy())

    def test(self):
        return self.model.predict(self.datamodule.test.data.numpy())

    def plot_evaluation(self, y_values: Dict[str, Tuple[np.ndarray, np.ndarray]], plotter: BasePlotter):
        # TODO
        pass
