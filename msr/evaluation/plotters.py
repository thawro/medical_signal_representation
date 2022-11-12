from abc import abstractmethod

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots

from msr.evaluation.visualisations import (
    matplotlib_confusion_matrix_plot,
    matplotlib_feature_importance_plot,
    plotly_confusion_matrix_plot,
    plotly_feature_importance_plot,
)

plotly_palette = px.colors.qualitative.Plotly
sns_palette = sns.color_palette()


class BasePlotter:
    @abstractmethod
    def roc_curve(self):
        pass

    @abstractmethod
    def confusion_matrix(self, y_values):
        pass

    @abstractmethod
    def feature_importances(self):
        pass

    @abstractmethod
    def target_vs_preds(self, y_values):
        pass

    @abstractmethod
    def metrics_comparison(self, metrics):
        pass


class MatplotlibPlotter(BasePlotter):
    def roc_curve(self, metrics, class_names):
        num_classes = len(class_names)
        fig, axes = plt.subplots(1, num_classes, figsize=(24, 4))
        for row, (split, split_metrics) in enumerate(metrics.items()):
            fprs, tprs, thresholds = split_metrics["roc"]
            for ax, fpr, tpr, class_name in zip(axes, fprs, tprs, class_names):
                ax.plot(fpr, tpr, color=sns_palette[row], ls="-", lw=2, label=split)
                ax.plot([0, 1], [0, 1], color="black", ls="--", lw=1)
                ax.set_title(class_name)
        plt.legend()
        return fig

    def confusion_matrix(self, y_values, class_names):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for ax, (split, split_y_values) in zip(axes, y_values.items()):
            matplotlib_confusion_matrix_plot(
                split_y_values["target"], split_y_values["preds"].argmax(axis=1), class_names=class_names, ax=ax
            )
            ax.set_title(split)
        return fig

    def feature_importances(self, feat_names, feat_importances, n_best=15, normalize=True):
        return matplotlib_feature_importance_plot(feat_names, feat_importances, n_best, normalize)

    def target_vs_preds(self, y_values):
        # TODO
        pass


class PlotlyPlotter(BasePlotter):
    def roc_curve(self, metrics, class_names):
        fig = make_subplots(rows=1, cols=len(class_names), subplot_titles=class_names)
        for row, (split, split_metrics) in enumerate(metrics.items()):
            fprs, tprs, thresholds = split_metrics["roc"]
            for col, (fpr, tpr, class_name) in enumerate(zip(fprs, tprs, class_names), start=1):
                fig.add_scatter(
                    showlegend=col == 1,
                    name=split,
                    marker=dict(color=plotly_palette[row]),
                    mode="lines",
                    x=fpr,
                    y=tpr,
                    row=1,
                    col=col,
                )
                fig.add_scatter(
                    showlegend=False,
                    mode="lines",
                    line_dash="dash",
                    x=[0, 1],
                    y=[0, 1],
                    row=1,
                    col=col,
                    marker=dict(color="black"),
                )
        fig.show()
        return fig

    def confusion_matrix(self, y_values, class_names):
        split_names = list(y_values.keys())
        fig = make_subplots(1, len(split_names), horizontal_spacing=0.15, subplot_titles=split_names)
        for i, (_, split_y_values) in enumerate(y_values.items()):
            plotly_confusion_matrix_plot(
                split_y_values["target"],
                split_y_values["preds"].argmax(axis=1),
                class_names=class_names,
                fig=fig,
                row=1,
                col=i + 1,
            )
        fig.show()
        return fig

    def feature_importances(self, feat_names, feat_importances, n_best=15, normalize=True):
        fig = plotly_feature_importance_plot(feat_names, feat_importances, n_best, normalize)
        fig.show()
        return fig

    @abstractmethod
    def metrics_comparison(self, metrics):
        n_metrics = len(list(metrics.values())[0])
        fig = make_subplots(1, n_metrics, subplot_titles=list(list(metrics.values())[0].keys()))

        for i, (split, split_metrics) in enumerate(metrics.items()):
            for col, (metric, value) in enumerate(split_metrics.items(), start=1):
                fig.add_trace(
                    trace=go.Bar(
                        name=split, marker=dict(color=plotly_palette[i]), showlegend=col == 1, x=["splits"], y=[value]
                    ),
                    row=1,
                    col=col,
                )
        fig.update_layout(barmode="group")
        fig.show()
        return fig

    def target_vs_preds(self, y_values):
        split_names = list(y_values.keys())
        fig = make_subplots(1, len(split_names), horizontal_spacing=0.15, subplot_titles=split_names)
        for col, (_, split_y_values) in enumerate(y_values.items(), start=1):
            target, preds = split_y_values["target"], split_y_values["preds"]
            fig.add_scatter(x=target, y=preds, mode="markers", row=1, col=col)
            line_bounds = [min(target), max(target)]
            fig.add_scatter(
                x=line_bounds,
                y=line_bounds,
                mode="lines",
                line=dict(dash="dash"),
                marker=dict(color="black"),
                row=1,
                col=col,
            )
        fig.show()
        return fig
