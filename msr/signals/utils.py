from typing import Dict, List, Type, Union

import numpy as np
import pandas as pd
from scipy.integrate import simps
from scipy.interpolate import PchipInterpolator, interp1d

from msr.utils import EPSILON

SIGNAL_FIG_PARAMS = dict(fig_size=(24, 4), label_size=12, annot_size=10)
BEAT_FIG_PARAMS = dict(
    fig_size=(6, 3), label_size=12, title_size=14, annot_size=10, marker_size=120, fill_alpha=0.2, legend_size=12
)


def parse_nested_feats(
    nested_feats: Dict[str, Dict[str, Union[float, int, str]]], sep: str = "__"
) -> Dict[str, Union[float, int, str]]:
    """Flatten nested features, i.e. keys are concatenated using specified separator.

    Args:
        nested_feats (Dict[str, Dict[Union[str, float]]]): Dict with nested keys.

    Returns:
        Dict[str, Union[float, int, str]]: Flattened dict.
    """
    feats_df = pd.json_normalize(nested_feats, sep=sep)
    feats = feats_df.to_dict(orient="records")[0]
    return {name: val for name, val in feats.items()}


def parse_feats_to_array(features):
    return np.array(list(parse_nested_feats(features).values()))


def get_outliers_mask(arr: np.ndarray, IQR_scale: float) -> np.ndarray:
    """Return outliers mask defined by IQR method.

    Args:
        arr (np.ndarray): Array with outliers.
        IQR_scale (float): Interquartile range scale.

    Returns:
        np.ndarray: Boolean mask, where False values indicate outliers.
    """
    return (arr >= np.mean(arr) + IQR_scale * np.std(arr)) | (arr <= np.mean(arr) - IQR_scale * np.std(arr))


def z_score(arr: np.ndarray) -> np.ndarray:
    """Return array with z_score normalization applied"""
    return (arr - np.mean(arr)) / np.std(arr)


def calculate_area(sig, fs, start, end, method="simps"):
    _sig = np.copy(sig)
    _sig = _sig[start : end + 1]
    _sig = _sig - min([_sig[0], _sig[-1]])

    if method == "trapz":
        return np.trapz(_sig, dx=1 / fs)
    elif method == "simps":
        return simps(_sig, dx=1 / fs)


def calculate_energy(sig, start, end):
    return sum(sig[start : end + 1] ** 2)


def calculate_slope(t, sig, start, end):
    return (sig[end] - sig[start]) / (t[end] - t[start] + EPSILON)


def get_windows(start, max_len, win_len, step):
    end = start + win_len
    windows = []
    while end < max_len:
        windows.append((start, end))
        start += step
        end += step
    windows.append((start, max_len))
    return windows


def moving_average(data: np.ndarray, winsize: int):
    extended_data = np.concatenate((np.ones(winsize) * data[0], data, np.ones(winsize) * data[-1]))
    win = np.ones(winsize) / winsize
    smoothed = np.convolve(extended_data, win, mode="same")
    return smoothed[winsize:-winsize]


def resample(old_time, old_values, new_time, kind="nearest"):
    interp_fn = interp1d(old_time, old_values, kind=kind, fill_value="extrapolate")
    return interp_fn(new_time)


def get_valid_beats_values_mask(beats, IQR_scale=1.5):
    beats_data = np.array([beat.data for beat in beats])
    beats_times = np.array([beat.time for beat in beats])

    mean_vals = beats_data.mean(axis=1)
    min_vals = beats_data.min(axis=1)
    max_vals = beats_data.max(axis=1)
    durations = np.array([t[-1] - t[0] for t in beats_times])

    mean_vals_mask = ~get_outliers_mask(mean_vals, IQR_scale=IQR_scale)
    min_vals_mask = ~get_outliers_mask(min_vals, IQR_scale=IQR_scale)
    max_vals_mask = ~get_outliers_mask(max_vals, IQR_scale=IQR_scale)
    duration_mask = ~get_outliers_mask(durations, IQR_scale=IQR_scale)

    return mean_vals_mask & duration_mask & min_vals_mask & max_vals_mask


def get_valid_beats_mask(beats: List, max_duration=1.1, IQR_scale=1.5):
    duration_mask = np.array([beat.duration <= max_duration for beat in beats])
    values_mask = get_valid_beats_values_mask(beats, IQR_scale=IQR_scale)
    return duration_mask & values_mask


def interpolate_to_new_time(time, data, new_time):
    interp_fn = PchipInterpolator(time, data)
    interpolated = interp_fn(new_time)
    return interpolated
