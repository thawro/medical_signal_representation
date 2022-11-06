import random

import numpy as np


def find_repeats(arr: np.ndarray, min_n_repeats: int):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index."""
    condition = np.concatenate((arr[:-1] == arr[1:], np.array([False])))
    d = np.diff(condition)
    (idx,) = d.nonzero()
    idx += 1
    if condition[0]:
        idx = np.r_[0, idx]
    if condition[-1]:
        idx = np.r_[idx, condition.size]  # Edit
    idx.shape = (-1, 2)
    return [segment.tolist() for segment in idx if (segment[1] - segment[0]) >= min_n_repeats - 1]


def validate_sig_values_range(data: np.ndarray, low: float = None, high: float = None):
    """Validate signal values in terms of magnitude.
    Returns list[bool] where True indicates samples within (low, high) range.
    """
    if low is None or high is None:
        return np.zeros_like(data) == 0  # since low and high bounds are None, all values are good
    else:
        return (data > low) & (data < high)  # True when data sample in range, False otherwise


def validate_sig_nan_values(data: np.ndarray):
    """Validate signal in terms of nan values.
    Returns list[bool] where False indicates nan samples.
    """
    return ~np.isnan(data)  # True when data sample is not nan, Flase otherwise


def validate_sig_repeat_values(data: np.ndarray, n_same: int):
    """Validate signal in terms of repeated values.
    Returns list[bool] where False indicates samples repeated for at least n_same times.
    """
    different_vals = np.zeros_like(data) == 0
    repeats_bounds = find_repeats(data, n_same)
    for start_idx, end_idx in repeats_bounds:
        different_vals[start_idx : end_idx + 1] = False
    return different_vals


def validate_signal(data: np.ndarray, low: float = None, high: float = None, max_n_same: int = 1):
    """Validate signal in terms of range, nan and repeated values.
    Returns list[bool] where True indicates samples which are good for all validation steps.
    """
    valid_range = validate_sig_values_range(data, low, high)
    valid_no_nans = validate_sig_nan_values(data)
    valid_different_vals = validate_sig_repeat_values(data, max_n_same)
    return valid_no_nans & valid_range & valid_different_vals


def cut_segments_into_samples(segments, n_samples, shuffle=True):
    """Split bigger segments bounds into smaller samples bounds"""
    samples_bounds = []
    for start, end in segments:
        segment_start, segment_end = start, start + n_samples
        while segment_end < end:
            samples_bounds.append((segment_start, segment_end))
            segment_start, segment_end = segment_end, segment_end + n_samples
    if shuffle:
        random.shuffle(samples_bounds)
    return samples_bounds
