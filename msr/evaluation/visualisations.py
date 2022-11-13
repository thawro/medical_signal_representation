import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix

palette = sns.color_palette()


def create_fig_if_axes_is_none(nrows=1, ncols=1, figsize=(7, 5), axes=None):
    if axes is None:
        return plt.subplots(nrows, ncols, figsize=figsize)
    else:
        return None, axes


def set_ax_params(ax, **kwargs):
    for name, value in kwargs.items():
        if value is None:
            pass
        elif isinstance(value, dict):
            getattr(ax, f"set_{name}")(**value)
        else:
            getattr(ax, f"set_{name}")(value)


def matplotlib_lineplot(x, y, ax=None, ls="--", lw=1, color=palette[0], **kwargs):
    fig, ax = create_fig_if_axes_is_none(axes=ax)
    ax.plot(x, y, color=color, ls=ls, lw=lw)
    set_ax_params(ax, **kwargs)
    return ax


def matplotlib_scatterplot(x, y, ax=None, marker="o", s=100, color=palette[0], **kwargs):
    fig, ax = create_fig_if_axes_is_none(axes=ax)
    ax.scatter(x, y, color=color, marker=marker, s=s)
    set_ax_params(ax, **kwargs)
    return ax


def matplotlib_roc_plot(fpr, tpr, class_name, ax=None):
    fig, ax = create_fig_if_axes_is_none(1, 1, figsize=(3, 5), axes=ax)
    matplotlib_lineplot(x=fpr, y=tpr, xlabel="FPR", ylabel="TPR", ls="-", title=class_name, ax=ax)
    matplotlib_lineplot(x=[0, 1], y=[0, 1], color="black", ls="--", ax=ax)
    return ax


def matplotlib_confusion_matrix_plot(target: np.ndarray, preds: np.ndarray, class_names: np.ndarray, ax=None):
    """Plot confustion matric

    Args:
        targets (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        class_names (np.ndarray): Class names.
        ax (_type_): Axes to plot on. If `None`, new Axes is created. Defaults to `None`.
    """
    fig, ax = create_fig_if_axes_is_none(1, 1, figsize=(16, 12), axes=ax)
    conf_matrix = pd.DataFrame(confusion_matrix(target, preds), columns=class_names, index=class_names).astype(int)
    sns.heatmap(conf_matrix, ax=ax, annot=True, fmt=".4g", cmap="Blues", cbar=False)
    set_ax_params(
        ax,
        xlabel=dict(xlabel="Predicted", fontsize=20),
        ylabel=dict(ylabel="True", fontsize=20),
        title=dict(label="Confusion Matrix", fontsize=22),
    )
    return fig


def matplotlib_feature_importance_plot(
    feat_names: list, feat_importances: list, n_best: int = 10, normalize: bool = True, ax=None
):
    """Plot feature importance

    Args:
        feat_names (list): Name of features passed in same order as importances.
        feat_importances (list): Importances of features passed in same order as feature names.
        n_best (int): Amount of best features to plot. Defaults to `10`.
        normalize (bool): Whether to normalize the importances. Defaults to `True`.
        ax (_type_): Axes to plot on. If `None`, new Axes is created. Defaults to `None`.
    """
    fig, ax = create_fig_if_axes_is_none(1, 1, figsize=(8, n_best / 1.8), axes=ax)
    sorted_importances = sorted(feat_importances, reverse=False)
    sorted_feat_names = [feat_name for _, feat_name in sorted(zip(feat_importances, feat_names), reverse=False)]
    best_feat_names = np.array(sorted_feat_names[-n_best:])
    normalize_factor = sum(feat_importances) if normalize else 1
    best_feat_importanes = np.array(sorted_importances[-n_best:]) / normalize_factor
    ax.barh(best_feat_names, best_feat_importanes)
    set_ax_params(
        ax,
        xlabel=dict(xlabel="Importance", fontsize=20),
        ylabel=dict(ylabel="Feature", fontsize=20),
        title=dict(label="Feature importance", fontsize=22),
    )
    return fig


def matplotlib_preds_vs_target(preds, target, ax=None):
    fig, ax = create_fig_if_axes_is_none(1, 1, figsize=(8, 8), axes=ax)
    matplotlib_scatterplot(target, preds, ax=ax)
    matplotlib_lineplot([min(target), max(target)], [min(target), max(target)], ls="--", color="black", ax=ax)
    set_ax_params(
        ax,
        xlabel=dict(xlabel="Target", fontsize=20),
        ylabel=dict(ylabel="Preds", fontsize=20),
        title=dict(label="Target vs Preds", fontsize=22),
    )


def plotly_lineplot(
    x=None,
    y=None,
    df=None,
    color=palette[0],
    line_dash=None,
    width=800,
    height=400,
    fig=None,
    row=None,
    col=None,
    **kwargs,
):
    color = f"rgb({f','.join([str(int(c*255)) for c in color])})" if isinstance(color, (tuple, list)) else color
    if fig is None:
        fig = px.line(data_frame=df, x=x, y=y, line_dash=line_dash, width=width, height=height, color=color)
    else:
        fig.add_scatter(mode="lines", x=x, y=y, row=row, col=col, line=dict(dash=line_dash), marker=dict(color=color))
    fig.update_layout(**kwargs)
    return fig


def plotly_scatterplot(
    x=None, y=None, df=None, color=palette[0], width=800, height=400, fig=None, row=None, col=None, **kwargs
):
    if fig is None:
        fig = px.scatter(data_frame=df, x=x, y=y, width=width, height=height)
    else:
        fig.add_scatter(mode="markers", x=x, y=y, row=row, col=col)
    fig.update_layout(**kwargs)
    return fig


def plotly_roc_plot(fpr, tpr, class_name, fig=None, row=None, col=None):
    if fig is None:
        fig = px.line(x=fpr, y=tpr)
    else:
        fig.add_scatter(name=class_name, mode="lines", x=fpr, y=tpr, row=row, col=col)
    fig.add_scatter(
        showlegend=False,
        mode="lines",
        line_dash="dash",
        x=[0, 1],
        y=[0, 1],
        row=row,
        col=col,
        marker=dict(color="black"),
    )
    return fig


def plotly_confusion_matrix_plot(
    target: np.ndarray, preds: np.ndarray, class_names: np.ndarray, fig=None, row=None, col=None
):
    conf_matrix = confusion_matrix(target, preds)
    subfig = ff.create_annotated_heatmap(
        conf_matrix, annotation_text=conf_matrix, colorscale="Blues", hoverinfo="z", x=class_names, y=class_names
    )
    subfig.update_yaxes(autorange="reversed")
    if fig is None:
        fig = subfig
    else:
        fig.add_trace(trace=go.Heatmap(subfig["data"][0]), row=row, col=col)
        for annotation in subfig.layout.annotations:
            fig.add_annotation(annotation, row=row, col=col)
        fig.update_yaxes(autorange="reversed")
    return fig


def plotly_feature_importance_plot(feat_names: list, feat_importances: list, n_best: int = 10, normalize: bool = True):
    sorted_importances = sorted(feat_importances, reverse=False)
    sorted_feat_names = [feat_name for _, feat_name in sorted(zip(feat_importances, feat_names), reverse=False)]
    best_feat_names = np.array(sorted_feat_names[-n_best:])
    normalize_factor = sum(feat_importances) if normalize else 1
    best_feat_importanes = np.array(sorted_importances[-n_best:]) / normalize_factor
    df = pd.DataFrame({"Feature name": best_feat_names, "Importance": best_feat_importanes})
    fig = px.bar(
        data_frame=df, x="Importance", y="Feature name", orientation="h", height=n_best * 50, title="Feature importance"
    )
    return fig
