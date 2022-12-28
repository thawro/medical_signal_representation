from collections import Counter

import numpy as np
import pywt
from scipy.stats import entropy, kurtosis, skew


def calculate_entropy(arr):
    counter_values = Counter(arr).most_common()
    probs = [el[1] / len(arr) for el in counter_values]
    return {"entropy": entropy(probs)}


def calculate_statistics(arr):
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
    }


def calculate_zerocross_feats(arr, thr=0):
    zerocross_idxs = np.nonzero(np.diff(arr > thr))[0]

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

    return {
        "cross_number": len(zerocross_idxs),
        "pos_cross_std": pos_zc_std,
        "neg_cross_std": neg_zc_std,
    }


def get_basic_signal_features(arr):
    entropy = calculate_entropy(arr)
    zerocross = calculate_zerocross_feats(arr, thr=0)
    zerocross = {f"zero_{name}": value for name, value in zerocross.items()}
    meancross = calculate_zerocross_feats(arr, thr=arr.mean())
    meancross = {f"mean_{name}": value for name, value in meancross.items()}

    statistics = calculate_statistics(arr)
    return {**entropy, **zerocross, **meancross, **statistics}


def extract_dwt_features(signal, wavelet="db5", level=None):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    level = len(coeffs) - 1
    names = [f"approximation_{level}"] + [f"detail_{level - i}" for i in range(level)]
    features = {}
    for name, coeff in zip(names, coeffs):
        features[name] = get_basic_signal_features(coeff)
    return features


def get_dwt_freqs_and_coeffs(time, data, wavelet="db5", level=None):
    if level is None:
        level = pywt.dwt_max_level(len(data), wavelet)
    dt = time[1] - time[0]
    coeffs = pywt.wavedec(data, wavelet, level=level)
    scales = [2 ** (level - 1)] + [2 ** (level - i) for i in range(1, level + 1)]
    freqs = np.array([pywt.scale2frequency(wavelet, scale) / dt for scale in scales])
    return freqs.round(4), coeffs


def extract_dwt_features(time, data, wavelet="db5", level=None):
    freqs, coeffs = get_dwt_freqs_and_coeffs(time, data, wavelet, level)
    level = len(freqs) - 1
    names = [f"approximation_{level}_freq_{freqs[0]}"] + [
        f"detail_{level - i}_freq_{freqs[i + 1]}" for i in range(level)
    ]
    features = {}
    for name, coeff in zip(names, coeffs):
        features[name] = get_basic_signal_features(coeff)
    return features


def get_cwt_img(time, data, freq_dim_scale=256, wavelet="morl", plot=True):
    """Continuous wavelet transform"""
    dt = time[1] - time[0]
    freq_centre = pywt.central_frequency(wavelet)
    cparam = 2 * freq_centre * freq_dim_scale
    scales = cparam / np.arange(1, freq_dim_scale + 1, 1)
    cwt_matrix, freqs = pywt.cwt(data, scales, wavelet, dt)
    cwt_matrix = abs(cwt_matrix)
    if plot:
        fig, ax = plt.subplots(figsize=(14, 8))
        im = ax.contourf(x, freqs, cwt_matrix, cmap=plt.cm.seismic)
        fig.colorbar(im)
    return freqs, cwt_matrix
