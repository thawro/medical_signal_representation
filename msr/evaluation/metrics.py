from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def get_classification_metrics(
    y_true: np.ndarray, y_pred_proba: np.ndarray, average: str = "macro", auc: bool = True
) -> Dict[str, float]:
    """Return classification metrics, i.e. F1-score, accuracy and AUC.

    Args:
        y_pred_proba (np.ndarray): Predicted class probabilities.
        y_true (np.ndarray): Ground truth values.
        average (str): This parameter is required for multiclass/multilabel targets.
            If ``None``, the scores for each class are returned.
            Otherwise, this determines the type of averaging performed on the data. Defaults to "macro".
        auc (bool): Whether to calculate AUC score. Defaults to `True`.

    Returns:
        Dict[str, float]: Dict with metrics.
    """
    y_pred = y_pred_proba.argmax(1)
    metrics = {
        "fscore": f1_score(y_true, y_pred, average=average),
        "acc": accuracy_score(y_true, y_pred),
    }
    if auc:
        metrics["auc"] = roc_auc_score(y_true, y_pred_proba, average=average, multi_class="ovr")
    return metrics


def get_regression_metrics(y_true, y_pred):
    # TODO
    pass
