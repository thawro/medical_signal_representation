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
    def plot_evaluation(self, y_values: Dict[str, Tuple[np.ndarray, np.ndarray]]):
        pass

    def evaluate(self, plot=False):
        y_pred_val = self.validate()
        y_pred_test = self.test()
        y_values = {
            "val": {"preds": y_pred_val, "target": self.datamodule.val.targets},
            "test": {"preds": y_pred_test, "target": self.datamodule.test.targets},
        }
        val_metrics = self.metrics.get_metrics(**y_values["val"])
        test_metrics = self.metrics.get_metrics(**y_values["test"])

        print(val_metrics)
        print(test_metrics)

        if plot:
            self.plot_evaluation(y_values)


class MLClassifierTrainer(MLTrainer):
    def __init__(self, model: BasePredictor, datamodule: BaseDataModule):
        self.model = model
        self.datamodule = datamodule
        self.class_names = datamodule.class_names
        self.feature_names = datamodule.feature_names
        self.metrics = ClafficationMetrics(num_classes=len(self.class_names))

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)

    def validate(self):
        return self.predict_proba(self.datamodule.val.data.numpy())

    def test(self):
        return self.predict_proba(self.datamodule.test.data.numpy())

    def plot_evaluation(self, y_values: Dict[str, Tuple[np.ndarray, np.ndarray]]):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for ax, (split, y_dict) in zip(axes, y_values.items()):
            plot_confusion_matrix(y_dict["target"], y_dict["preds"].argmax(axis=1), class_names=self.class_names, ax=ax)
            ax.set_title(split)

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
