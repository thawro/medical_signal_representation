from collections import OrderedDict
from typing import Dict

import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import seaborn as sns
from biosppy.signals.ecg import ecg
from scipy.signal import find_peaks

from signals.base import BeatSignal, MultiChannelPeriodicSignal, PeriodicSignal
from signals.utils import (
    calculate_area,
    calculate_energy,
    calculate_slope,
    create_new_obj,
    lazy_property,
    parse_feats_to_array,
    parse_nested_feats,
)

CHANNELS_POLARITY = [1, 1, -1, -1, 1, 1, -1, -1, -1, 1, 1, 1]  # some channels are with oposite polarity


def check_ecg_polarity(data):
    sig = data - data.mean()
    pos_peaks, _ = find_peaks(sig)
    neg_peaks, _ = find_peaks(-sig)
    pos_peaks_amps = sig[pos_peaks]
    neg_peaks_amps = abs(sig[neg_peaks])
    pos_amp = np.quantile(pos_peaks_amps, 0.95)
    neg_amp = np.quantile(neg_peaks_amps, 0.95)
    if pos_amp >= neg_amp:
        return 1
    # print("Wrong polarity")
    return -1


def create_multichannel_ecg(data, fs):
    signals = OrderedDict({i + 1: ECGSignal(f"ECG_{i+1}", channel_data, fs) for i, channel_data in enumerate(data)})
    return MultiChannelECGSignal(signals)


class ECGSignal(PeriodicSignal):
    def __init__(self, name, data, fs, start_sec=0):
        polarity = check_ecg_polarity(nk.ecg_clean(data, sampling_rate=fs))
        super().__init__(name, polarity * data, fs, start_sec)
        self.cleaned = nk.ecg_clean(self.data, sampling_rate=self.fs)

    @lazy_property
    def rpeaks(self):
        return nk.ecg_peaks(self.cleaned, sampling_rate=self.fs)[1]["ECG_R_Peaks"]

    def get_beats(
        self, intervals=None, resample=True, n_samples=100, validate=True, plot=False, use_raw=False, return_arr=False
    ):
        if intervals is None:
            qrs_epochs = nk.ecg_segment(self.cleaned, rpeaks=None, sampling_rate=self.fs, show=False)
            beats_times = []
            intervals = []
            prev_end_idx = qrs_epochs["1"].Index.values[-1]
            for idx, beat in qrs_epochs.items():
                if int(idx) == 1:
                    beats_times.append(beat.index.values)
                else:
                    start_idx = np.where(beat.Index.values == prev_end_idx)[0]
                    start_idx = start_idx[0] if len(start_idx) == 1 else 0
                    beat = beat.iloc[start_idx:]
                    beats_times.append(beat.index.values)
                if beat.Index.values[1] >= self.n_samples or beat.Index.values[0] >= self.n_samples:
                    break
                prev_end_idx = beat.Index.values[-1]
                intervals.append([beat.Index.values[0], beat.Index.values[-1]])

            avg_start_time = np.median([beat[0] for beat in beats_times[1:]])
            first_beat_mask = beats_times[0] > avg_start_time
            beats_times[0] = beats_times[0][first_beat_mask]
            intervals[0][0] = intervals[0][0] + (first_beat_mask == False).sum()
            if intervals[-1][1] >= self.n_samples:
                intervals[-1][1] = self.n_samples - 1
            if intervals[0][0] < 0:
                intervals[0][0] = 0
            intervals = [tuple(interval) for interval in intervals]
        self.beats_intervals = intervals
        self.valid_beats_mask = len(intervals) * [True]
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

    def extract_hrv_features(self, return_arr=True):
        if self.rpeaks is None:
            self.find_rpeaks()
        r_peaks = self.rpeaks
        r_vals = self.data[r_peaks]
        r_times = self.time[r_peaks]
        ibi = np.diff(r_times)

        features = OrderedDict({"ibi_mean": np.mean(ibi), "ibi_std": np.std(ibi), "R_val": np.mean(r_vals)})
        if return_arr:
            return parse_feats_to_array(features)
        return features

    def extract_per_beat_features(self, plot=False, return_arr=True, n_beats=-1, **kwargs):
        if self.beats is None:
            self.get_beats(**kwargs)
        n_beats = len(self.beats) if n_beats <= 0 else n_beats
        beats = self.beats[:n_beats]

        if return_arr:
            return np.array([beat.extract_features(plot=plot, return_arr=True) for beat in beats])
        return OrderedDict(
            {f"beat_{beat.beat_num}": beat.extract_features(plot=plot, return_arr=False) for beat in beats}
        )

    def extract_features(self, return_arr=True, feats_to_extract=["basic", "hrv", "agg_beat"], plot=False):
        if self.agg_beat is None:
            self.aggregate()
        features = OrderedDict({"whole_signal_features": {}})
        if "basic" in feats_to_extract:
            features["whole_signal_features"] = super().extract_features(return_arr=False)
        if "hrv" in feats_to_extract:
            features["whole_signal_features"]["hrv_features"] = self.extract_hrv_features(return_arr=False)
        if "agg_beat" in feats_to_extract:
            features["agg_beat_features"] = self.agg_beat.extract_features(plot=plot, return_arr=False)
        if "per_beat" in feats_to_extract:
            features["per_beat_features"] = self.extract_per_beat_features(plot=plot, return_arr=False)
        if return_arr:
            return parse_feats_to_array(features)
        return features


class ECGBeat(BeatSignal):
    def __init__(self, name, data, fs, start_sec, end_sec, beat_num=0):
        super().__init__(name, data, fs, start_sec, end_sec, beat_num)

    @lazy_property
    def r_loc(self):
        peaks, _ = find_peaks(self.data)
        peaks_vals = self.data[peaks]
        r_vals = iter(np.flip(np.sort(peaks_vals)))
        r_loc = peaks[np.where(peaks_vals == next(r_vals))[0][0]]
        while r_loc / self.n_samples >= 0.7:  # find first largest peak which occurs below 70% of total time
            next_largest_peak = np.where(peaks_vals == next(r_vals))[0][0]
            r_loc = peaks[next_largest_peak]
        return r_loc

    @lazy_property
    def p_loc(self):
        try:
            zero_to_r_data = self.data[: self.r_loc]
            peaks, _ = find_peaks(zero_to_r_data)
            biggest_peak_idx = np.array([zero_to_r_data[peak] for peak in peaks]).argmax()
            return peaks[biggest_peak_idx]  # zero_to_r_data.argmax()
        except Exception:
            # TODO
            return int(self.r_loc / 4)

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
        try:
            s_to_end_data = self.data[self.s_loc :]
            peaks, _ = find_peaks(s_to_end_data)
            biggest_peak_idx = np.array([s_to_end_data[peak] for peak in peaks]).argmax()
            return peaks[biggest_peak_idx] + self.s_loc  # zero_to_r_data.argmax()
        except Exception:
            # TODO
            return self.s_loc + (self.n_samples - self.s_loc) // 2

    @lazy_property
    def crit_points(self):
        return {"p": self.p_loc, "q": self.q_loc, "r": self.r_loc, "s": self.s_loc, "t": self.t_loc}

    def extract_crit_points_features(self, return_arr=True, plot=False, ax=None):
        features = OrderedDict()
        for name, loc in self.crit_points.items():
            features[f"{name}_loc"] = loc
            features[f"{name}_time"] = self.time[loc]
            features[f"{name}_val"] = self.data[loc]
        if plot:
            self.plot(ax=ax, with_crit_points=True)
        if return_arr:
            return parse_feats_to_array(features)
        return features

    def extract_area_features(self, return_arr=True, plot=False, ax=None):
        dx = 1 / self.fs
        params = [
            {"name": "ZeroPQ_A", "start": 0, "end": self.q_loc},
            {"name": "QRS_A", "start": self.q_loc, "end": self.s_loc},
            {"name": "STEnd_A", "start": self.s_loc, "end": self.n_samples - 1},
        ]
        features = OrderedDict({p["name"]: calculate_area(self.data, dx, p["start"], p["end"]) for p in params})
        if plot:
            if ax is None:
                fig, ax = plt.subplots(figsize=(14, 6))
            palette = sns.color_palette("pastel", len(params))
            self.plot(ax=ax)
            ax.get_legend().remove()
            areas = []
            for i, p in enumerate(params):
                mask = (np.arange(self.n_samples) >= p["start"]) & (np.arange(self.n_samples) <= p["end"])
                a = ax.fill_between(
                    self.time[mask],
                    self.data[mask],
                    alpha=0.4,
                    color=palette[i],
                    edgecolor="black",
                    lw=1,
                    label=p["name"],
                )
                areas.append(a)
            leg = ax.legend(handles=areas, fontsize=14)
            ax.add_artist(leg)
        if return_arr:
            return parse_feats_to_array(features)
        return features

    def extract_energy_features(self, return_arr=True):
        params = [
            {"name": "ZeroPQ_E", "start": 0, "end": self.q_loc},
            {"name": "QRS_E", "start": self.q_loc, "end": self.s_loc},
            {"name": "STEnd_E", "start": self.s_loc, "end": self.n_samples - 1},
        ]

        features = OrderedDict({p["name"]: calculate_energy(self.data, p["start"], p["end"]) for p in params})
        if return_arr:
            return parse_feats_to_array(features)
        return features

    def extract_slope_features(self, return_arr=True):
        params = [
            {"name": "PQ_slope", "start": self.p_loc, "end": self.q_loc},
            {"name": "QR_slope", "start": self.q_loc, "end": self.r_loc},
            {"name": "RS_slope", "start": self.r_loc, "end": self.s_loc},
            {"name": "ST_slope", "start": self.s_loc, "end": self.t_loc},
        ]

        features = OrderedDict({p["name"]: calculate_slope(self.time, self.data, p["start"], p["end"]) for p in params})
        if return_arr:
            return parse_feats_to_array(features)
        return features

    def extract_features(self, return_arr=True, plot=False):
        features = super().extract_features(return_arr=False)
        features["crit_points_features"] = self.extract_crit_points_features(return_arr=False)
        features["area_features"] = self.extract_area_features(return_arr=False)
        features["energy_features"] = self.extract_energy_features(return_arr=False)
        features["slope_features"] = self.extract_slope_features(return_arr=False)
        if return_arr:
            return parse_feats_to_array(features)
        return features


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

    def plot_beats_segmentation(self, use_raw=False, **kwargs):
        self.get_beats(**kwargs)
        fig, axes = plt.subplots(self.n_signals, 1, figsize=(24, 1 * self.n_signals), sharex=True)
        for ax, (sig_name, sig) in zip(axes, self.signals.items()):
            sig.plot_beats_segmentation(use_raw=use_raw, ax=ax)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title("")
            ax.grid(False)
            ax.get_legend().remove()

    def get_beats(self, source_channel=2, resample=True, return_arr=True, **kwargs):
        if source_channel is not None:
            self.signals[source_channel].get_beats(**kwargs)
            source_channel_intervals = self.signals[source_channel].beats_intervals
        else:
            source_channel_intervals = None
        all_channels_beats = [
            signal.get_beats(source_channel_intervals, n_samples=100) for _, signal in self.signals.items()
        ]
        beats_times = [beat.time for beat in all_channels_beats[0]]
        if return_arr:
            all_channels_beats = [[beat.data for beat in channel_beats] for channel_beats in all_channels_beats]
        self.beats = np.array(all_channels_beats)
        self.beats_times = np.array(beats_times)
        return self.beats

    def get_waveform_representation(self, return_arr=True):
        if return_arr:
            return np.array([sig.data for name, sig in self.signals.items()])
        else:
            return OrderedDict({name: sig.data for name, sig in self.signals.items()})

    def get_per_beat_features_representation(self, return_arr=True, **kwargs):
        self.get_beats(**kwargs)
        if return_arr:
            return np.array([sig.extract_per_beat_features(return_arr=True) for _, sig in self.signals.items()])
        return OrderedDict(
            {name: sig.extract_per_beat_features(return_arr=False) for name, sig in self.signals.items()}
        )

    def get_agg_beat_features_representation(self, return_arr=True, **kwargs):
        self.get_beats(**kwargs)
        agg_beats = OrderedDict({name: signal.aggregate() for name, signal in self.signals.items()})
        if return_arr:
            return np.array([agg_beat.extract_features(return_arr=True) for _, agg_beat in agg_beats.items()])
        return OrderedDict({name: agg_beat.extract_features(return_arr=False) for name, agg_beat in agg_beats.items()})

    def get_whole_signal_features_representation(self, return_arr=True, **kwargs):
        self.get_beats(**kwargs)
        if return_arr:
            return np.array([sig.extract_features(return_arr=True) for _, sig in self.signals.items()])
        return OrderedDict({name: sig.extract_features(return_arr=False) for name, sig in self.signals.items()})
