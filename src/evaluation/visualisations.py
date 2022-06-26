import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def plot_confusion_matrix(y_true: list, y_pred: list, class_names: list, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 12))
    accuracy = accuracy_score(y_true, y_pred)
    fscore = f1_score(y_true, y_pred, average="macro")
    conf_matrix = pd.DataFrame(confusion_matrix(y_true, y_pred), columns=class_names, index=class_names).astype(int)
    sns.heatmap(conf_matrix, ax=ax, annot=True, fmt=".3g", cmap="Blues")
    ax.set_xlabel("Predicted", fontsize=20)
    ax.set_ylabel("True", fontsize=20)
    ax.set_title(
        f"Confusion Matrix (accuracy = {accuracy:.2f}, fscore = {fscore:.2f})",
        fontsize=22,
    )


def plot_feature_importance(
    feat_names: list,
    feat_importances: list,
    n_best: int = 10,
    normalize: bool = True,
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, n_best / 1.8))
    sorted_importances = sorted(feat_importances, reverse=False)
    sorted_feat_names = [feat_name for _, feat_name in sorted(zip(feat_importances, feat_names), reverse=False)]
    best_feat_names = np.array(sorted_feat_names[-n_best:])
    normalize_factor = sum(feat_importances) if normalize else 1
    best_feat_importanes = np.array(sorted_importances[-n_best:]) / normalize_factor
    ax.barh(best_feat_names, best_feat_importanes)
    ax.set_title("Feature importance", fontsize=22)
