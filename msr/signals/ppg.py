from collections import OrderedDict
from typing import List, Type

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import integrate
from scipy.signal import butter, filtfilt

from msr.signals.base import BeatSignal, PeriodicSignal
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

    z_mean = squared.mean()  # 7
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
        self.feature_extraction_funcs.update(
            {
                "hrv_features": self.extract_hrv_features,
            }
        )

    @property
    def BeatClass(self):
        return PPGBeat

    def find_troughs(self):
        peaks = zip(self.peaks[:-1], self.peaks[1:])
        return np.array([self.data[prev_peak:next_peak].argmin() + prev_peak for prev_peak, next_peak in peaks])

    def find_peaks(self):
        peaks = find_systolic_peaks_ELGENDI(self.data, self.fs)
        return peaks

    def _get_beats_intervals(self):
        intervals = np.vstack((self.troughs[:-1], self.troughs[1:])).T
        return intervals

    def extract_hrv_features(self, return_arr=False, plot=False):
        ibi = np.diff(self.time[self.peaks])
        features = OrderedDict({"ibi_mean": np.mean(ibi), "ibi_std": np.std(ibi)})
        if return_arr:
            return parse_feats_to_array(features)
        return features

    def extract_agg_beat_features(self, return_arr=True, plot=False):
        return self.agg_beat.extract_features(plot=plot, return_arr=return_arr)

    def explore(self, start_time=0, width=None, window_size=4, min_hz=0, max_hz=20):
        super().explore(start_time, width, window_size, min_hz, max_hz)


class PPGBeat(BeatSignal):
    def __init__(self, name, data, fs, start_sec, beat_num=0):
        super().__init__(name, data, fs, start_sec, beat_num)
        self.feature_extraction_funcs.update(
            {
                "sppg_features": self.extract_sppg_features,
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

    def extract_sppg_features(self, return_arr=False, plot=True):
        systolic_onset_slope = (self.systolic_peak_val - self.data[0]) / self.systolic_peak_time
        systolic_offset_slope = (self.systolic_peak_val - self.data[-1]) / (self.duration - self.systolic_peak_time)
        energy = (self.data**2).mean()
        before_systolic_area = integrate.simpson(
            abs(self.data[: self.systolic_peak_loc]), self.time[: self.systolic_peak_loc]
        )
        after_systolic_area = integrate.simpson(
            abs(self.data[self.systolic_peak_loc :]), self.time[self.systolic_peak_loc :]
        )

        if plot:
            fig, ax = plt.subplots(figsize=self.fig_params["fig_size"])
            ax.plot(self.time, self.data, c=sns.color_palette()[0], lw=3)
            ax.scatter(
                self.time[self.systolic_peak_loc],
                self.data[self.systolic_peak_loc],
                s=self.fig_params["marker_size"],
                c="r",
                marker="^",
                label="systolic peak",
            )
            ax.fill_between(
                self.time,
                self.data,
                self.data.min(),
                where=self.time <= self.time[self.systolic_peak_loc],
                color="g",
                alpha=self.fig_params["fill_alpha"],
            )
            ax.fill_between(
                self.time,
                self.data,
                self.data.min(),
                where=self.time >= self.time[self.systolic_peak_loc],
                color="r",
                alpha=self.fig_params["fill_alpha"],
            )
            ax.set_title(f"sPPG", fontsize=self.fig_params["title_size"])
            ax.set_xlabel("Time [s]", fontsize=self.fig_params["label_size"])
            ax.set_ylabel("Values [a.u.]", fontsize=self.fig_params["label_size"])
            ax.legend()

        features = OrderedDict(
            {
                "duration": self.duration,
                "systolic_peak_val": self.systolic_peak_val,
                "systolic_peak_time": self.systolic_peak_time,
                "systolic_onset_slope": systolic_onset_slope,
                "systolic_offset_slope": systolic_offset_slope,
                "energy": energy,
                "before_systolic_area": before_systolic_area,
                "after_systolic_area": after_systolic_area,
            }
        )
        if return_arr:
            return parse_feats_to_array(features)
        return features

    def explore(self, start_time=0, width=None, window_size=4, min_hz=0, max_hz=20):
        super().explore(start_time, width, window_size, min_hz, max_hz)
