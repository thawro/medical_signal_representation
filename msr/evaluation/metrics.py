from typing import Dict, List

import numpy as np
import torch
from torchmetrics.functional import (
    accuracy,
    auc,
    auroc,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    pearson_corrcoef,
    r2_score,
    roc,
)


def get_metrics(metrics_funcs, preds: torch.Tensor, target: torch.Tensor):
    preds = torch.from_numpy(preds) if isinstance(preds, np.ndarray) else preds
    target = torch.from_numpy(target) if isinstance(target, np.ndarray) else target
    metrics = {}
    for name, metric in metrics_funcs.items():
        if name in ["auc"]:  # AUC requires both tensors to be 1D
            metrics[name] = metric(preds.argmax(1), target).item()
        elif name in ["roc"]:
            fpr, tpr, threshold = metric(preds, target)
            metrics[name] = fpr, tpr, threshold
        else:
            metrics[name] = metric(preds, target).item()
    return metrics


def get_classification_metrics(num_classes, preds, target):
    classification_metrics = dict(
        accuracy=accuracy(num_classes=num_classes),
        fscore=f1_score(num_classes=num_classes, average="macro"),
        auroc=auroc(num_classes=num_classes),
        auc=auc(reorder=True),
        roc=roc(num_classes=num_classes),
    )
    return get_metrics(metrics_funcs=classification_metrics, preds=preds, target=target)


def get_regression_metrics(preds, target):
    regression_metrics = dict(
        mae=mean_absolute_error(),
        mape=mean_absolute_percentage_error(),
        corr=pearson_corrcoef(),
        r2=r2_score(),
    )
    return get_metrics(metrics_funcs=regression_metrics, preds=preds, target=target)
