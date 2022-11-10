from abc import abstractmethod
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from msr.evaluation.metrics import (
    ClafficationMetrics,
    RegressionMetrics,
    get_classification_metrics,
    get_regression_metrics,
)
from msr.evaluation.visualisations import plot_confusion_matrix, plot_feature_importance
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
    def plot_evaluation(self, y_values: Dict[str, Tuple[np.ndarray, np.ndarray]], metrics: Dict[str, float]):
        pass

    def evaluate(self, plot=False):
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

        if plot:
            self.plot_evaluation(y_values, metrics)

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

    def plot_roc(self, fpr, tpr, class_names, axes=None):
        num_classes = len(class_names)
        if axes is None:
            fig, axes = plt.subplots(1, num_classes, figsize=(num_classes * 3, 5))
        for ax, _fpr, _tpr, class_name in zip(axes, fpr, tpr, class_names):
            ax.plot(_fpr, _tpr)
            ax.plot([0, 1], [0, 1], ls="--", color="black", lw=0.6)
            ax.set_title(class_name)

    def plot_evaluation(self, y_values: Dict[str, Tuple[np.ndarray, np.ndarray]], metrics: Dict[str, float]):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for ax, (split, split_y_values) in zip(axes, y_values.items()):
            plot_confusion_matrix(
                split_y_values["target"], split_y_values["preds"].argmax(axis=1), class_names=self.class_names, ax=ax
            )
            ax.set_title(split)

        for split, split_metrics in metrics.items():
            fig, axes = plt.subplots(1, self.num_classes, figsize=(self.num_classes * 3.5, 3))
            fprs, tprs, thresholds = split_metrics["roc"]
            self.plot_roc(fprs, tprs, self.class_names, axes)
            fig.suptitle(split)

        if hasattr(self.model, "feature_importances_"):
            plot_feature_importance(
                feat_names=self.feature_names, feat_importances=self.model.feature_importances_, n_best=15
            )


class MLRegressorTrainer(MLTrainer):
    def __init__(self, model: BasePredictor, datamodule: BaseDataModule):
        self.model = model
        self.datamodule = datamodule
        self.metrics = RegressionMetrics()

    def validate(self):
        return self.model.predict(self.datamodule.val.data.numpy())

    def test(self):
        return self.model.predict(self.datamodule.test.data.numpy())

    def plot_evaluation(self, y_values: Dict[str, Tuple[np.ndarray, np.ndarray]]):
        # TODO
        pass
