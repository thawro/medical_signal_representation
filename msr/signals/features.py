from collections import Counter
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.stats import entropy, kurtosis, skew

from msr.signals.utils import SIGNAL_FIG_PARAMS


def calculate_entropy_features(arr):
    counter_values = Counter(arr).most_common()
    probs = [el[1] / len(arr) for el in counter_values]
    return {"entropy": entropy(probs)}


def calculate_statistic_features(arr):
    calculate_rms = lambda arr: np.mean(np.sqrt(arr**2))
    calculate_energy = lambda arr: sum(arr**2)
    calculate_rms = lambda arr: sum(abs(arr))

    percentiles = [5, 25, 75, 95]
    get_stat_or_nan = lambda func, arr: func(arr) if len(arr) > 0 else np.nan
    statistic_funcs = {
        "mean": np.mean,
        "median": np.median,
        "std": np.std,
        "var": np.var,
        "kurtosis": kurtosis,
        "skew": skew,
        **{f"percentile_{q}": partial(np.percentile, q=q) for q in percentiles},
        "root_mean_square": calculate_rms,
        "energy": calculate_energy,
        "auc": calculate_rms,
    }
    return {name: get_stat_or_nan(func, arr) for name, func in statistic_funcs.items()}


def calculate_zerocross_features(arr, thr=0):
    zerocross_idxs = np.nonzero(np.diff(arr > thr))[0]
    if len(zerocross_idxs) == 0:
        num_crosses = 0
        pos_cross_std = -1
        neg_cross_std = -1
    elif len(zerocross_idxs) <= 2:
        num_crosses = len(zerocross_idxs)
        pos_cross_std = 0
        neg_cross_std = 0
    else:
        first_zc_idx = max(0, zerocross_idxs[0] - 1)
        if arr[first_zc_idx + 1] - arr[first_zc_idx] > 0:  # first zc is from minus to plus
            pos_zc_idxs = zerocross_idxs[::2]
            neg_zc_idxs = zerocross_idxs[1::2]
        else:
            neg_zc_idxs = zerocross_idxs[::2]
            pos_zc_idxs = zerocross_idxs[1::2]

        neg_zc_diffs = np.diff(neg_zc_idxs)
        pos_zc_diffs = np.diff(pos_zc_idxs)

        neg_zc_std = neg_zc_diffs.std()
        pos_zc_std = pos_zc_diffs.std()

        num_crosses = len(zerocross_idxs)
        pos_cross_std = pos_zc_std
        neg_cross_std = neg_zc_std

    return {
        "num_crosses": num_crosses,
        "pos_cross_std": pos_cross_std,
        "neg_cross_std": neg_cross_std,
    }


def get_basic_signal_features(arr):
    entropy = calculate_entropy_features(arr)
    zerocross = calculate_zerocross_features(arr, thr=0)
    zerocross = {f"zero_{name}": value for name, value in zerocross.items()}
    meancross = calculate_zerocross_features(arr, thr=np.mean(arr))
    meancross = {f"mean_{name}": value for name, value in meancross.items()}
    statistics = calculate_statistic_features(arr)
    return {**entropy, **zerocross, **meancross, **statistics}


def get_dwt_coeffs(data, fs, wavelet="db8", level=None):
    approximates = []
    details = []
    dt = 1 / fs
    if level is None:
        level = pywt.dwt_max_level(len(data), wavelet)

    approx_data = data.copy()
    scale = 1
    for i in range(level):
        (approx_data, detail) = pywt.dwt(approx_data, wavelet)
        freq = pywt.scale2frequency(wavelet, scale) / dt
        freq = round(freq, 2)
        details.append({"freq": (freq / 2, freq), "data": detail})
        approximates.append({"freq": (0, freq / 2), "data": approx_data})
        scale *= 2
    return approximates, details


def plot_dwt(data, approximates, details):
    level = len(details)
    w, h = SIGNAL_FIG_PARAMS["fig_size"]
    h = int(0.3 * h * len(details))
    w = w // 3
    fig = plt.figure(figsize=(w, h), layout="constrained")
    spec = fig.add_gridspec(level + 1, 2)

    ax0 = fig.add_subplot(spec[0, :])
    ax0.tick_params(axis="both", which="major", labelsize=SIGNAL_FIG_PARAMS["tick_size"] // 2)
    all_axes = np.array([[fig.add_subplot(spec[i + 1, j]) for j in range(2)] for i in range(level)])

    # fig.suptitle("Discrete wavelet transform", fontsize=SIGNAL_FIG_PARAMS['title_size'])
    ax0.plot(data, lw=3)
    ax0.set_title("Original signal", fontsize=SIGNAL_FIG_PARAMS["title_size"] // 3)
    for i, (axes, detail, approx) in enumerate(zip(all_axes, details, approximates)):
        approx_label = f"{' - '.join([str(round(f, 1)) for f in approx['freq']])} Hz"
        detail_label = f"{' - '.join([str(round(f, 1)) for f in detail['freq']])} Hz"
        axes[0].plot(approx["data"], "r", label=approx_label)
        axes[1].plot(detail["data"], "g", label=detail_label)
        axes[0].set_ylabel(f"Level {i+1}", fontsize=SIGNAL_FIG_PARAMS["label_size"] // 2, rotation=90)
        # axes[0].set_yticklabels([])
        # axes[0].legend(fontsize=SIGNAL_FIG_PARAMS["tick_size"] // 2)
        # axes[1].legend(fontsize=SIGNAL_FIG_PARAMS["tick_size"] // 2)
        if i == 0:
            approx_title = "Approximation coefficients"
            detail_title = "Detail coefficients"
            axes[0].set_title(approx_title, fontsize=SIGNAL_FIG_PARAMS["title_size"] // 3)
            axes[1].set_title(detail_title, fontsize=SIGNAL_FIG_PARAMS["title_size"] // 3)

        # axes[1].set_yticklabels([])

    for ax in all_axes.flatten():
        ax.tick_params(axis="both", which="major", labelsize=SIGNAL_FIG_PARAMS["tick_size"] // 3)

    all_axes[-1][0].set_xlabel("Sample [-]", fontsize=SIGNAL_FIG_PARAMS["label_size"] // 2)
    all_axes[-1][1].set_xlabel("Sample [-]", fontsize=SIGNAL_FIG_PARAMS["label_size"] // 2)
    plt.tight_layout()

    return fig


def extract_dwt_features(signal, wavelet="db8", level=None, plot=False):
    data, fs = signal.cleaned, signal.fs
    approximates, details = get_dwt_coeffs(data, fs, wavelet, level)
    if plot:
        fig = plot_dwt(data, approximates, details)
    last_approx = approximates[-1]
    features = {}
    energies = []
    for i, detail in enumerate(details):
        feats = get_basic_signal_features(detail["data"])
        energies.append(feats["energy"])
        features[f"detail_{i+1}_freq_{detail['freq']}"] = feats
    probs = np.array(energies) / sum(energies)
    features[f"approx_{len(details)}_freq_{last_approx['freq']}"] = get_basic_signal_features(last_approx["data"])
    features["wavelet_entropy"] = entropy(probs)
    if plot:
        return features, fig
    return features


def get_cwt_img(data, fs, freq_dim_scale=256, wavelet="morl"):
    """Continuous wavelet transform"""
    dt = 1 / fs
    freq_centre = pywt.central_frequency(wavelet)
    cparam = 2 * freq_centre * freq_dim_scale
    scales = cparam / np.arange(1, freq_dim_scale + 1, 1)
    cwt_matrix, freqs = pywt.cwt(data, scales, wavelet, dt)
    cwt_matrix = abs(cwt_matrix)
    return freqs, cwt_matrix
