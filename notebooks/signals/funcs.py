import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from msr.signals.utils import BEAT_FIG_PARAMS

sns.set(style="whitegrid")


def plot_raw_signal(signal):
    # fig, axes = plt.subplots(1, 1, figsize=signal.fig_params["fig_size"])
    fig = signal.plot(0, -1, label=None, title=None)
    plt.close()
    return fig


def plot_beat_signal(signal):
    beat = signal.agg_beat
    # fig, axes = plt.subplots(1, 1, figsize=beat.fig_params["fig_size"])
    fig = beat.plot(0, -1, label=None, title=None)
    plt.close()
    return fig


def plot_signal_and_agg_beat(signal):
    fig = signal.plot_signal_and_agg_beat()
    plt.close()
    return fig


def plot_beat_segmentation(signal):
    fig = signal.plot_beats_segmentation()
    plt.close()
    return fig


def plot_raw_vs_cleaned(signal):
    fig, axes = plt.subplots(1, 1, figsize=signal.fig_params["fig_size"])
    axes.plot(signal.time, signal.data, lw=3, label="Raw")
    axes.plot(signal.time, signal.cleaned, lw=3, label="Cleaned")
    axes.set_xlabel("Time [s]", fontsize=signal.fig_params["label_size"])
    axes.set_ylabel(signal.units, fontsize=signal.fig_params["label_size"])
    plt.legend(fontsize=signal.fig_params["legend_size"])
    plt.close()
    return fig


def plot_peaks_troughs_features(signal, ax=None):
    return_fig = False
    if ax is None:
        return_fig = True
        fig, ax = plt.subplots(figsize=signal.fig_params["fig_size"])
    feats = signal.extract_peaks_troughs_features(return_arr=False, plot=True, ax=ax)
    ax.set_ylabel(signal.units, fontsize=signal.fig_params["label_size"])
    ax.set_title("Peaks troughs signals", fontsize=signal.fig_params["title_size"])
    plt.close()
    if return_fig:
        return fig


def plot_ecg_features(signal):
    fig, axes = plt.subplots(1, 2, figsize=signal.fig_params["fig_size"], gridspec_kw={"width_ratios": [8, 3]})
    plot_peaks_troughs_features(signal, ax=axes[0])
    plot_ecg_beat_features(signal, ax=axes[1])
    axes[1].set_ylabel("")
    plt.tight_layout()
    return fig


def plot_ppg_features(signal):
    w, h = signal.fig_params["fig_size"]
    fig = plt.figure(figsize=(w, h * 2), layout="constrained")
    spec = fig.add_gridspec(2, 3)
    ax0 = fig.add_subplot(spec[0, :])
    ax0.tick_params(axis="both", which="major", labelsize=signal.fig_params["tick_size"])
    plot_peaks_troughs_features(signal, ax=ax0)
    axes = np.array([fig.add_subplot(spec[1, j]) for j in range(3)])
    axes[1].set_title("")
    axes[2].set_title("")
    plot_ppg_beat_features(signal, axes=axes)
    return fig


def plot_eeg_features(signal):
    fig, axes = plt.subplots(1, 2, figsize=signal.fig_params["fig_size"], gridspec_kw={"width_ratios": [8, 3]})
    plot_peaks_troughs_features(signal, ax=axes[0])
    signal.psd(plot=True, fmin=0, fmax=50, ax=axes[1])
    axes[1].set_title("Power spectral density", fontsize=BEAT_FIG_PARAMS["title_size"])
    plt.tight_layout()
    return fig


def plot_eog_features(signal):
    fig, ax = plt.subplots(1, 1, figsize=signal.fig_params["fig_size"])

    # fig, axes = plt.subplots(1, 2, figsize=signal.fig_params["fig_size"], gridspec_kw={"width_ratios": [8, 3]})
    # plot_peaks_troughs_features(signal, ax=axes[0])
    plot_peaks_troughs_features(signal, ax=ax)

    # TODO
    # plot_ecg_beat_features(signal, ax=axes[1])
    # axes[1].set_ylabel("")
    plt.tight_layout()
    return fig


def plot_DWT_features(signal):
    feats, fig = signal.extract_DWT_features(return_arr=False, plot=True)
    plt.close()
    return fig


def plot_ecg_beat_features(signal, crit_points=["p", "q", "r", "s", "t"], ax=None):
    beat = signal.agg_beat
    return_fig = False
    if ax is None:
        return_fig = True
        fig, ax = plt.subplots(1, 1, figsize=beat.fig_params["fig_size"])
    beat.plot_area_features(ax=ax, crit_points=crit_points)
    ax.set_title("Area features", fontsize=beat.fig_params["title_size"])
    plt.close()
    if return_fig:
        return fig


def plot_ppg_beat_features(signal, crit_points=["systolic_peak"], axes=None):
    signal.set_beats()
    signal.set_agg_beat()
    beat = signal.agg_beat
    return_fig = False
    if axes is None:
        return_fig = True
        ncols = 3
        figsize = beat.fig_params["fig_size"]
        figsize = figsize[0] * ncols, figsize[1]
        fig, axes = plt.subplots(1, ncols, figsize=figsize)
    beat.plot_area_features(ax=axes[0], crit_points=crit_points)
    pw_feats = beat.extract_pulse_width_features(return_arr=False, plot=True, ax=axes[1])
    ph_feats = beat.extract_pulse_height_features(return_arr=False, plot=True, ax=axes[2])
    axes[0].set_title("Area features", fontsize=beat.fig_params["title_size"])
    axes[1].set_title("Pulse width features", fontsize=beat.fig_params["title_size"])
    axes[2].set_title("Pulse height features", fontsize=beat.fig_params["title_size"])
    plt.close()
    if return_fig:
        return fig


def get_figs_for_thesis_periodic(signal, prefix="ecg_"):
    signal.set_beats()
    signal.set_agg_beat()
    figs = {
        f"{prefix}signal": plot_raw_signal(signal),
        f"{prefix}cleaning": plot_raw_vs_cleaned(signal),
        # f"{prefix}beat_signal": plot_beat_signal(signal),
        f"{prefix}signal_and_agg_beat": plot_signal_and_agg_beat(signal),
        f"{prefix}beats_segmentation": plot_beat_segmentation(signal),
        # f"{prefix}beat_features": plot_ecg_beat_features(signal) if prefix == "ecg_" else plot_ppg_beat_features(signal),
        # f"{prefix}peaks_troughs_features": plot_peaks_troughs_features(signal),
        f"{prefix}DWT_features": plot_DWT_features(signal),
    }
    if prefix == "ecg_":
        figs[f"{prefix}features"] = plot_ecg_features(signal)
    elif prefix == "ppg_":
        figs[f"{prefix}features"] = plot_ppg_features(signal)
    return figs


def get_figs_for_thesis_no_periodic(signal, prefix="eeg_"):
    figs = {
        f"{prefix}signal": plot_raw_signal(signal),
        f"{prefix}cleaning": plot_raw_vs_cleaned(signal),
        # f"{prefix}peaks_troughs_features": plot_peaks_troughs_features(signal),
        f"{prefix}DWT_features": plot_DWT_features(signal),
    }
    if prefix == "eeg_":
        figs[f"{prefix}features"] = plot_eeg_features(signal)
    elif prefix == "eog_":
        figs[f"{prefix}features"] = plot_eog_features(signal)
    return figs
