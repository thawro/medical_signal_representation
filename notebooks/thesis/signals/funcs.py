import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set(style="whitegrid")


def plot_raw_signal(signal):
    fig, axes = plt.subplots(1, 1, figsize=signal.fig_params["fig_size"])
    axes.plot(signal.time, signal.data, lw=3, label="Raw")
    axes.set_xlabel("Time [s]", fontsize=signal.fig_params["label_size"])
    axes.set_ylabel(signal.units, fontsize=signal.fig_params["label_size"])
    plt.legend(fontsize=signal.fig_params["legend_size"])
    plt.close()
    return fig


def plot_beat_signal(signal):
    beat = signal.agg_beat
    fig, axes = plt.subplots(1, 1, figsize=beat.fig_params["fig_size"])
    axes.plot(beat.time, beat.data, lw=3, label="Single ECG beat")
    axes.set_xlabel("Time [s]", fontsize=beat.fig_params["label_size"])
    axes.set_ylabel(signal.units, fontsize=beat.fig_params["label_size"])
    plt.legend(fontsize=beat.fig_params["legend_size"])
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


def plot_peaks_troughs_features(signal):
    fig, ax = plt.subplots(figsize=signal.fig_params["fig_size"])
    feats = signal.extract_peaks_troughs_features(return_arr=False, plot=True, ax=ax)
    ax.set_ylabel(signal.units, fontsize=signal.fig_params["label_size"])
    ax.set_title("")
    plt.close()
    return fig


def plot_DWT_features(signal):
    feats, fig = signal.extract_DWT_features(return_arr=False, plot=True)
    plt.close()
    return fig


def plot_ecg_beat_features(signal, crit_points=["p", "q", "r", "s", "t"]):
    beat = signal.agg_beat
    fig, axes = plt.subplots(1, 1, figsize=beat.fig_params["fig_size"])
    beat.plot_area_features(ax=axes, crit_points=crit_points)
    axes.set_title("Area features", fontsize=beat.fig_params["title_size"])
    plt.close()
    return fig


def plot_ppg_beat_features(signal, crit_points=["systolic_peak"]):
    signal.set_beats()
    signal.set_agg_beat()
    beat = signal.agg_beat
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
    return fig


def get_figs_for_thesis_periodic(signal, prefix="ecg_"):
    signal.set_beats()
    signal.set_agg_beat()
    figs = {
        f"{prefix}signal": plot_raw_signal(signal),
        f"{prefix}cleaning": plot_raw_vs_cleaned(signal),
        f"{prefix}beat_signal": plot_beat_signal(signal),
        f"{prefix}beats_segmentation": plot_beat_segmentation(signal),
        f"{prefix}beat_features": plot_ecg_beat_features(signal)
        if prefix == "ecg_"
        else plot_ppg_beat_features(signal),
        f"{prefix}peaks_troughs_features": plot_peaks_troughs_features(signal),
        f"{prefix}DWT_features": plot_DWT_features(signal),
    }
    return figs


def get_figs_for_thesis_no_periodic(signal, prefix="eeg_"):
    figs = {
        f"{prefix}signal": plot_raw_signal(signal),
        f"{prefix}cleaning": plot_raw_vs_cleaned(signal),
        f"{prefix}peaks_troughs_features": plot_peaks_troughs_features(signal),
        f"{prefix}DWT_features": plot_DWT_features(signal),
    }
    return figs
