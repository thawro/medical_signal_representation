from abc import abstractmethod

import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

from msr.evaluation.visualisations import (
    matplotlib_confusion_matrix_plot,
    matplotlib_feature_importance_plot,
    matplotlib_roc_plot,
    plotly_confusion_matrix_plot,
    plotly_feature_importance_plot,
    plotly_roc_plot,
)


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


class MatplotlibPlotter(BasePlotter):
    def roc_curve(self, metrics, class_names):
        num_classes = len(class_names)
        for split, split_metrics in metrics.items():
            fig, axes = plt.subplots(1, num_classes, figsize=(num_classes * 3.5, 3))
            fprs, tprs, thresholds = split_metrics["roc"]
            matplotlib_roc_plot(fprs, tprs, class_names, axes=axes)
            fig.suptitle(split)
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
        figs = []
        for split, split_metrics in metrics.items():
            fprs, tprs, thresholds = split_metrics["roc"]
            fig = plotly_roc_plot(fprs, tprs, class_names)
            fig.update_layout(title=split)
            fig.show()
            figs.append(fig)
        return figs

    def confusion_matrix(self, y_values, class_names):
        split_names = list(y_values.keys())
        fig = make_subplots(1, 2, horizontal_spacing=0.15, subplot_titles=split_names)
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

    def target_vs_preds(self, y_values):
        # TODO
        pass
