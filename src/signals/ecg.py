from typing import Dict

import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
from biosppy.signals.ecg import ecg
from scipy.signal import find_peaks

from data.ptbxl import CHANNELS_POLARITY
from signals.base import BeatSignal, MultiChannelPeriodicSignal, PeriodicSignal
from signals.utils import lazy_property


def create_multichannel_ecg(data, fs):
    signals = {
        i + 1: ECGSignal(f"ECG_{i+1}", channel_data * CHANNELS_POLARITY[i], fs) for i, channel_data in enumerate(data)
    }
    return MultiChannelECGSignal(signals)


class ECGSignal(PeriodicSignal):
    def __init__(self, name, data, fs, start_sec=0):
        super().__init__(name, data, fs, start_sec)
        self.cleaned = nk.ecg_clean(self.data, sampling_rate=self.fs)

    @lazy_property
    def rpeaks(self):
        return nk.ecg_peaks(self.cleaned, sampling_rate=self.fs)[1]["ECG_R_Peaks"]

    def get_beats(
        self, intervals=None, resample=True, n_samples=100, validate=True, plot=False, use_raw=True, return_arr=False
    ):
        self.valid_beats_mask = len(self.beats) * [True]
        if intervals is None:
            qrs_epochs = nk.ecg_segment(self.cleaned, rpeaks=None, sampling_rate=self.fs, show=False)
            beats_times = []
            beats_data = []
            intervals = []
            prev_end_idx = qrs_epochs["1"].Index.values[-1]
            for idx, beat in qrs_epochs.items():
                idx = int(idx)
                if idx == 1:
                    beats_times.append(beat.index.values)
                    beats_data.append(beat.Signal.values)
                else:
                    start_idx = np.where(beat.Index.values == prev_end_idx)[0]
                    if len(start_idx) == 1:
                        start_idx = start_idx[0]
                    elif len(start_idx) == 0:
                        start_idx = 0
                    beat = beat.iloc[start_idx:]
                    beats_times.append(beat.index.values)
                    beats_data.append(beat.Signal.values)
                prev_end_idx = beat.Index.values[-1]
                intervals.append([beat.Index.values[0], beat.Index.values[-1]])

            avg_start_time = np.median([beat[0] for beat in beats_times[1:]])
            first_beat_mask = beats_times[0] > avg_start_time
            beats_times[0] = beats_times[0][first_beat_mask]
            beats_data[0] = beats_data[0][first_beat_mask]
            intervals[0][0] = intervals[0][0] + (first_beat_mask == False).sum()
            intervals = [tuple(interval) for interval in intervals]

        data_source = self.data if use_raw else self.cleaned
        beats = []
        for i, interval in enumerate(intervals):
            start_sec, end_sec = interval[0] / self.fs, interval[1] / self.fs
            beat_data = data_source[interval[0] : interval[1] + 1]
            beat = ECGBeat(
                name=f"beat {i}", data=beat_data, fs=self.fs, start_sec=start_sec, end_sec=end_sec, beat_num=i
            )
            if resample:
                beat = beat.resample_with_interpolation(n_samples=100, kind="pchip")
            beats.append(beat)
        self.beats = np.array(beats)
        if plot:
            self.plot_beats()
        if return_arr:
            beats_times = np.array([beat.time for beat in self.beats])
            beats_data = np.array([beat.data for beat in self.beats])
            return beats_times, beats_data
        return self.beats

    def extract_ecg_features(self):
        if self.rpeaks is None:
            self.find_rpeaks()
        r_peaks = self.rpeaks
        r_vals = self.data[r_peaks]
        r_times = self.time[r_peaks]
        ibi = np.diff(r_times)

        self.ecg_features = {"ibi_mean": np.mean(ibi), "ibi_std": np.std(ibi), "R_val": np.mean(r_vals)}

        return self.ecg_features

    def extract_features(self, plot=True):
        single_ecg = self.aggregate()
        self.features = super().extract_features()
        ecg_features = {"ecg_features": self.extract_ecg_features()}
        agg_beat_features = {"agg_beat_features": single_ecg.extract_agg_beat_features(plot=plot)}
        self.features.update(ecg_features)
        self.features.update(agg_beat_features)
        return self.features


class ECGBeat(BeatSignal):
    def __init__(self, name, data, fs, start_sec, end_sec, beat_num=0):
        super().__init__(name, data, fs, start_sec, end_sec, beat_num)

    @lazy_property
    def r_loc(self):
        return self.data.argmax()

    @lazy_property
    def p_loc(self):
        zero_to_r_data = self.data[: self.r_loc]
        peaks, _ = find_peaks(zero_to_r_data)
        biggest_peak_idx = np.array([zero_to_r_data[peak] for peak in peaks]).argmax()
        return peaks[biggest_peak_idx]  # zero_to_r_data.argmax()

    @lazy_property
    def q_loc(self):
        p_to_r_data = self.data[self.p_loc : self.r_loc]
        return p_to_r_data.argmin() + self.p_loc

    @lazy_property
    def s_loc(self):
        r_to_end_data = self.data[self.r_loc :]
        troughs, _ = find_peaks(-r_to_end_data)
        return troughs[0] + self.r_loc

    @lazy_property
    def t_loc(self):
        s_to_end_data = self.data[self.s_loc :]
        peaks, _ = find_peaks(s_to_end_data)
        biggest_peak_idx = np.array([s_to_end_data[peak] for peak in peaks]).argmax()
        return peaks[biggest_peak_idx] + self.s_loc  # zero_to_r_data.argmax()

    @lazy_property
    def crit_points(self):
        return {"p": self.p_loc, "q": self.q_loc, "r": self.r_loc, "s": self.s_loc, "t": self.t_loc}

    def extract_agg_beat_features(self, plot=True, ax=None):
        feats = {}
        for name, loc in self.crit_points.items():
            feats[f"{name}_loc"] = loc
            feats[f"{name}_time"] = self.time[loc]
            feats[f"{name}_val"] = self.data[loc]
        if plot:
            self.plot(ax=ax, with_crit_points=True)
        return feats


class MultiChannelECGSignal(MultiChannelPeriodicSignal):
    def plot(self, **kwargs):
        fig, axes = plt.subplots(self.n_signals, 1, figsize=(24, 1 * self.n_signals), sharex=True)
        for ax, (sig_name, sig) in zip(axes, self.signals.items()):
            sig.plot(ax=ax, **kwargs)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title("")
            ax.grid(False)
            ax.get_legend().remove()

    def plot_beats_segmentation(self, **kwargs):
        fig, axes = plt.subplots(self.n_signals, 1, figsize=(24, 1 * self.n_signals), sharex=True)
        for ax, (sig_name, sig) in zip(axes, self.signals.items()):
            sig.plot_beats_segmentation(ax=ax, **kwargs)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title("")
            ax.grid(False)
            ax.get_legend().remove()

    def get_beats(self, source_channel=2, **kwargs):
        source_channel_beats = signals[source_channel].get_beats(**kwargs)

        self.beats = {name: signal.get_beats(**kwargs) for name, signal in self.signals.items()}
        return self.beats

    def aggregate(self, **kwargs):
        self.agg_beat = {name: signal.aggregate(**kwargs) for name, signal in self.signals.items()}
        return self.agg_beat
