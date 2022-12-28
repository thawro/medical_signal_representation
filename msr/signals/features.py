from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.stats import entropy, kurtosis, skew


def calculate_entropy_features(arr):
    counter_values = Counter(arr).most_common()
    probs = [el[1] / len(arr) for el in counter_values]
    return {"entropy": entropy(probs)}


def calculate_statistic_features(arr):
    return {
        "percentile_5": np.percentile(arr, 5),
        "percentile_25": np.percentile(arr, 25),
        "percentile_75": np.percentile(arr, 75),
        "percentile_95": np.percentile(arr, 95),
        "median": np.median(arr),
        "mean": np.mean(arr),
        "std": np.std(arr),
        "var": np.var(arr),
        "root_mean_square": np.mean(np.sqrt(arr**2)),
        "kurtosis": kurtosis(arr),
        "skew": skew(arr),
        "energy": sum(arr**2),
        "auc": sum(abs(arr)),
    }


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


def get_dwt_coeffs(data, fs, wavelet="db5", level=None):
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
    fig = plt.figure(figsize=(10, 13), layout="constrained")
    spec = fig.add_gridspec(level + 1, 2)

    ax0 = fig.add_subplot(spec[0, :])

    all_axes = [[fig.add_subplot(spec[i + 1, j]) for j in range(2)] for i in range(level)]

    fig.suptitle("Discrete wavelet transform", fontsize=18)
    ax0.plot(data)
    for i, (axes, detail, approx) in enumerate(zip(all_axes, details, approximates)):
        axes[0].plot(approx["data"], "r")
        axes[1].plot(detail["data"], "g")
        axes[0].set_ylabel(f"Level {i+1}", fontsize=12, rotation=90)
        # axes[0].set_yticklabels([])
        approx_title = f"{' - '.join([str(round(f, 2)) for f in approx['freq']])} Hz"
        detail_title = f"{' - '.join([str(round(f, 2)) for f in detail['freq']])} Hz"
        if i == 0:
            approx_title = "Approximation coefficients" + f"\n{approx_title}"
            detail_title = "Detail coefficients" + f"\n{detail_title}"
        axes[0].set_title(approx_title, fontsize=12)
        axes[1].set_title(detail_title, fontsize=12)
        # axes[1].set_yticklabels([])


def extract_dwt_features(data, fs, wavelet="db5", level=None, plot=False):
    approximates, details = get_dwt_coeffs(data, fs, wavelet, level)
    if plot:
        plot_dwt(data, approximates, details)
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
