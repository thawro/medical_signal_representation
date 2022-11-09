from abc import abstractmethod
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from msr.evaluation.metrics import get_classification_metrics, get_regression_metrics
from msr.evaluation.visualisations import plot_confusion_matrix, plot_feature_importance
from msr.training.data.datamodules import BaseDataModule
from msr.training.utils import BasePredictor


class MLTrainer:
    def __init__(self, model: BasePredictor):
        self.model = model

    def fit(self, datamodule: BaseDataModule):
        self.model.fit(X=datamodule.train.data.numpy(), y=datamodule.train.targets)

    @abstractmethod
    def get_metrics(self, *args):
        pass

    @abstractmethod
    def validate(self, datamodule: BaseDataModule):
        pass

    @abstractmethod
    def test(self, datamodule: BaseDataModule):
        pass

    @abstractmethod
    def plot_evaluation(self, y_values: Dict[str, Tuple[np.ndarray, np.ndarray]]):
        pass

    def evaluate(self, datamodule: BaseDataModule, plot=False):
        y_pred_val = self.validate(datamodule)
        y_pred_test = self.test(datamodule)
        y_values = {
            "val": (datamodule.val.targets, y_pred_val),
            "test": (datamodule.test.targets, y_pred_test),
        }
        val_metrics = self.get_metrics(*y_values["val"])
        test_metrics = self.get_metrics(*y_values["test"])

        print(val_metrics)
        print(test_metrics)

        if plot:
            self.plot_evaluation(y_values, datamodule)


class MLClassifierTrainer(MLTrainer):
    def get_metrics(self, *args):
        return get_classification_metrics(*args)

    def validate(self, datamodule: BaseDataModule):
        return self.model.predict_proba(datamodule.val.data.numpy())

    def test(self, datamodule: BaseDataModule):
        return self.model.predict_proba(datamodule.test.data.numpy())

    def plot_evaluation(self, y_values: Dict[str, Tuple[np.ndarray, np.ndarray]], datamodule: BaseDataModule):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for ax, (split, (y_true, y_pred)) in zip(axes, y_values.items()):
            plot_confusion_matrix(y_true, y_pred.argmax(axis=1), class_names=datamodule.class_names, ax=ax)
            ax.set_title(split)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for ax, (split, (y_true, y_pred)) in zip(axes, y_values.items()):
            plot_feature_importance(
                feat_names=datamodule.feature_names, feat_importances=self.model.feature_importances_, n_best=15, ax=ax
            )
            ax.set_title(split)


class MLRegressorTrainer(MLTrainer):
    def get_metrics(self, *args):
        return get_regression_metrics(*args)

    def validate(self, datamodule: BaseDataModule):
        return self.model.predict(datamodule.val.data.numpy())

    def test(self, datamodule: BaseDataModule):
        return self.model.predict(datamodule.test.data.numpy())

    def plot_evaluation(self, y_values: Dict[str, Tuple[np.ndarray, np.ndarray]], datamodule: BaseDataModule):
        pass
