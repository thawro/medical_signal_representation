from typing import Dict, List, Tuple, Type, Union

import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import seaborn as sns
from scipy import integrate
from scipy.signal import butter, filtfilt

from msr.signals.base import BeatSignal, PeriodicSignal, find_intervals_using_hr
from msr.signals.utils import moving_average, parse_feats_to_array
from msr.utils import lazy_property


def find_systolic_peaks_ELGENDI(
    data: np.ndarray, fs: float, f1: float = 0.5, f2: float = 8, w1: float = 0.111, w2: float = 0.667, beta: float = 0.2
):
    # http://dx.doi.org/10.1371/journal.pone.0076585
    cutoff = [f1, f2]
    cutoff = 2.0 * np.array(cutoff) / fs
    b, a = butter(N=2, Wn=cutoff, btype="bandpass", analog=False, output="ba")
    filtered = filtfilt(b, a, data)  # 2
    clipped = np.copy(filtered)  # 3
    clipped[clipped < 0] = 0
    squared = clipped**2  # 4

    ma_peak = moving_average(squared, int(w1 * fs))  # 5
    ma_beat = moving_average(squared, int(w2 * fs))  # 6

    z_mean = np.mean(squared)  # 7
    alpha = beta * z_mean  # 8

    thr_1 = ma_beat + alpha  # 9

    #  10 - 16
    valid_segments = ma_peak > thr_1
    starts = np.where(~valid_segments[0:-1] & valid_segments[1:])[0]
    ends = np.where(valid_segments[0:-1] & ~valid_segments[1:])[0]
    ends = ends[ends > starts[0]]

    blocks = zip(starts, ends)  # 17
    thr_2 = w1  # 18

    peaks = []
    #  19 - 25
    for start, end in blocks:
        if end - start >= thr_2:
            peaks.append(data[start:end].argmax() + start)
    return np.array(peaks)


class PPGSignal(PeriodicSignal):
    def __init__(self, name: str, data: np.ndarray, fs: float, start_sec: float = 0):
        super().__init__(name, data, fs, start_sec)
        self.units = "PPG [a.u.]"
        # self.nk_signals_df, self.nk_info = nk.ppg_process(self.data, sampling_rate=self.fs)
        self.feature_extraction_funcs.update(
            {
                "hrv": self.extract_hrv_features,
            }
        )

    @property
    def cleaned(self):
        return nk.ppg_clean(self.data, sampling_rate=self.fs)

    @property
    def BeatClass(self):
        return PPGBeat

    def find_troughs(self):
        peaks = zip(self.peaks[:-1], self.peaks[1:])
        return np.array([self.data[prev_peak:next_peak].argmin() + prev_peak for prev_peak, next_peak in peaks])

    def find_peaks(self):
        peaks = find_systolic_peaks_ELGENDI(self.data, self.fs)
        # peaks = self.nk_info['PPG_Peaks'].values
        return peaks

    def _get_beats_intervals(self):
        try:
            intervals = np.vstack((self.troughs[:-1], self.troughs[1:])).T
        except Exception:
            intervals = find_intervals_using_hr(self)
        return intervals

    def extract_hrv_features(self, return_arr=False, plot=False):
        """
        source: https://neuropsychology.github.io/NeuroKit/functions/ppg.html#ppg-intervalrelated
        """
        peaks = self.peaks
        if len(peaks) == 0:
            hr, ibi_mean, ibi_std, mean_val = np.nan, np.nan, np.nan, np.nan
        else:
            vals = self.data[peaks]
            times = self.time[peaks]
            ibi = np.diff(times)
            ibi_mean = np.mean(ibi)
            ibi_std = np.std(ibi)
            mean_val = np.mean(vals)
            hr = self.duration / len(peaks) * 60
        features = {"hr": hr, "ibi_mean": ibi_mean, "ibi_std": ibi_std, "R_val": mean_val}
        # df = nk.ppg_intervalrelated(self.nk_signals_df, sampling_rate=self.fs)
        # features = df.to_dict(orient="records")[0]
        if return_arr:
            return parse_feats_to_array(features)
        return features

    def extract_agg_beat_features(self, return_arr=True, plot=False):
        return self.agg_beat.extract_features(plot=plot, return_arr=return_arr)

    def explore(self, start_time=0, width=None):
        figs = super().explore(start_time, width)
        return figs


class PPGBeat(BeatSignal):
    def __init__(self, name, data, fs, start_sec, beat_num=0):
        super().__init__(name, data, fs, start_sec, beat_num)
        self.units = "PPG [a.u.]"
        self.feature_extraction_funcs.update(
            {
                "pulse_width": self.extract_pulse_width_features,
                "pulse_height": self.extract_pulse_height_features,
            }
        )

        self.find_systolic_peak()

    def find_systolic_peak(self):
        self.systolic_peak_loc = self.data.argmax()
        if self.systolic_peak_loc == 0:
            self.systolic_peak_loc = self.n_samples // 2
        self.systolic_peak_time = self.time[self.systolic_peak_loc]
        self.systolic_peak_val = self.data[self.systolic_peak_loc]

    @lazy_property
    def crit_points(self):
        return {
            "systolic_peak": self.systolic_peak_loc,
        }

    @lazy_property
    def energy_features_crit_points(self) -> List[Dict[str, Union[int, str]]]:
        return [
            {"name": "ZeroSysE", "start": 0, "end": self.systolic_peak_loc},
            {"name": "SysEndE", "start": self.systolic_peak_loc, "end": self.n_samples - 1},
        ]

    @lazy_property
    def area_features_crit_points(self) -> List[Dict[str, Union[int, str]]]:
        """Must return list of dicts with `name`, `start` and `end` keys

        Example: [{"name": <feat_name>, "start": <start_location>, "end": <end_location>}, ...]
        """
        return [
            {"name": "ZeroSysA", "start": 0, "end": self.systolic_peak_loc},
            {"name": "SysEndA", "start": self.systolic_peak_loc, "end": self.n_samples - 1},
        ]

    @lazy_property
    def slope_features_crit_points(self) -> List[Dict[str, Union[int, str]]]:
        """Must return list of dicts with `name`, `start` and `end` keys

        Example: [{"name": <feat_name>, "start": <start_location>, "end": <end_location>}, ...]
        """
        return [
            {"name": "SysOnsetSlope", "start": 0, "end": self.systolic_peak_loc},
            {"name": "SysOffsetSlope", "start": self.systolic_peak_loc, "end": self.n_samples - 1},
        ]

    def extract_pulse_width_features(self, return_arr=False, height_pcts=[10, 25, 33, 50, 66, 75], plot=False, ax=None):
        """https://ieeexplore.ieee.org/document/9558767"""
        heights = np.array([pct / 100 * (self.range) + self.min for pct in height_pcts])
        idxs = [np.argwhere(np.diff(np.sign(height - self.data))).flatten() for height in heights]
        idxs = [idx if len(idx) >= 2 else [idx[0], self.n_samples - 1] for idx in idxs]
        widths = np.array([idx[-1] - idx[0] for idx in idxs])
        times = {height_pcts[i]: [self.time[idxs[i][0]], self.time[idxs[i][-1]]] for i in range(len(heights))}
        features = {f"pulse_width_{height_pct}%": width / self.fs for height_pct, width in zip(height_pcts, widths)}
        return_fig = False
        if plot:
            if ax is None:
                return_fig = True
                fig, ax = plt.subplots(figsize=self.fig_params["fig_size"])
            # ax.plot(self.time, self.data, "-", lw=3)
            self.plot(ax=ax, title="pulse width features", label="")
            for height in height_pcts:
                label = f"pw at {height}%"
                start, end = times[height]
                height_val = height / 100 * self.range + self.min
                ax.hlines(height_val, start, end, ls="--", color="black")
                ax.annotate(label, (end, height_val), size=self.fig_params["annot_size"])
        if return_arr:
            return parse_feats_to_array(features)
        if return_fig:
            return features, fig
        return features

    def extract_pulse_height_features(self, return_arr=False, width_pcts=[10, 25, 33, 50, 66, 75], plot=False, ax=None):
        """https://ieeexplore.ieee.org/document/9558767"""
        width_sample_locs = [int(self.n_samples * pct / 100) for pct in width_pcts]
        height_vals = [self.data[loc] - self.min for loc in width_sample_locs]
        features = {f"pulse_height_{width_pct}%": height_val for width_pct, height_val in zip(width_pcts, height_vals)}
        height_vals = {
            width_sample_loc: [self.min, self.data[width_sample_loc]] for width_sample_loc in width_sample_locs
        }
        return_fig = False
        if plot:
            if ax is None:
                return_fig = True
                fig, ax = plt.subplots(figsize=self.fig_params["fig_size"])
            self.plot(ax=ax, title="pulse height features", label="")
            for i, (width_loc, height_bounds) in enumerate(height_vals.items()):
                label = f"ph at {width_pcts[i]}%"
                lower, upper = height_bounds
                ax.vlines(width_loc / self.fs, lower, upper, ls="--", color="black")
                ax.annotate(label, (width_loc / self.fs, upper), size=self.fig_params["annot_size"])
        if return_arr:
            return parse_feats_to_array(features)
        if return_fig:
            return features, fig
        return features

    def explore(self, start_time=0, width=None):
        figs = super().explore(start_time, width)
        return figs
