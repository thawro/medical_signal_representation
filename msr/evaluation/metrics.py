from functools import partial
from typing import Callable, Dict, List

import numpy as np
import torch
from torchmetrics.functional import (
    accuracy,
    auroc,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    pearson_corrcoef,
    r2_score,
    roc,
)


def get_metrics(metrics_funcs: Dict[str, Callable], preds: torch.Tensor, target: torch.Tensor, metrics: List[str]):
    preds = torch.from_numpy(preds) if isinstance(preds, np.ndarray) else preds
    target = torch.from_numpy(target) if isinstance(target, np.ndarray) else target
    calculated_metrics = {}
    for name, metric in metrics_funcs.items():
        if name not in metrics:
            continue
        if name in ["roc"]:
            fpr, tpr, threshold = metric(preds, target)
            calculated_metrics[name] = fpr, tpr, threshold
        else:
            calculated_metrics[name] = metric(preds, target).item()
    return calculated_metrics


def get_classification_metrics(num_classes, preds, target, metrics):
    classification_metrics = dict(
        accuracy=partial(accuracy, num_classes=num_classes),
        fscore=partial(f1_score, num_classes=num_classes, average="weighted"),
        auroc=partial(auroc, num_classes=num_classes, average="weighted"),
        roc=partial(roc, num_classes=num_classes),
    )
    return get_metrics(metrics_funcs=classification_metrics, preds=preds, target=target, metrics=metrics)


def get_regression_metrics(preds, target, metrics):
    regression_metrics = dict(
        mae=mean_absolute_error,
        mape=mean_absolute_percentage_error,
        corr=pearson_corrcoef,
        r2=r2_score,
        mse=mean_squared_error,
    )
    return get_metrics(metrics_funcs=regression_metrics, preds=preds, target=target, metrics=metrics)
