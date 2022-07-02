import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
from biosppy.signals.ecg import ecg
from scipy.signal import find_peaks

from signals.base import BeatSignal, PeriodicSignal
from signals.utils import lazy_property


def get_ecg_beats(ecg_sig, fs=100, return_raw=True):
    clean_ecg = nk.ecg_clean(ecg_sig, sampling_rate=fs)
    qrs_epochs = nk.ecg_segment(clean_ecg, rpeaks=None, sampling_rate=fs, show=False)
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
            start_idx = np.where(beat.Index.values == prev_end_idx)[0][0]
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
    if return_raw:
        beats_data = [ecg_sig[start_idx : end_idx + 1] for start_idx, end_idx in intervals]
    return intervals, beats_times, beats_data


class ECGSignal(PeriodicSignal):
    def __init__(self, name, data, fs, start_sec=0):
        super().__init__(name, data, fs, start_sec)
        self.cleaned = nk.ecg_clean(self.data, sampling_rate=self.fs)

    @lazy_property
    def rpeaks(self):
        return nk.ecg_peaks(self.cleaned, sampling_rate=self.fs)[1]["ECG_R_Peaks"]

    def get_beats(self, resample_beats=True, n_samples=100, validate=True, plot=False, use_raw=True):
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
                start_idx = np.where(beat.Index.values == prev_end_idx)[0][0]
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
        if use_raw:
            beats_data = [self.data[start_idx : end_idx + 1] for start_idx, end_idx in intervals]
        beats = []
        for i, (interval, beat_time, beat_data) in enumerate(zip(intervals, beats_times, beats_data)):
            beats.append(
                ECGBeat(name=f"beat {i}", data=beat_data, fs=self.fs, start_sec=interval[0] * self.fs, beat_num=i)
            )
        self.beats = beats
        return self.beats

    def aggregate(self, valid_only=True, **kwargs):
        # agreguje listę uderzeń w pojedyncze uderzenie sPPG i je zwraca (jako obiekt PPGBeat)
        if self.beats is None:
            self.get_beats(**kwargs)
        beats_to_aggregate_mask = self.valid_beats_mask if valid_only else len(self.beats) * [True]
        if beats_to_aggregate_mask is None:
            beats_to_aggregate_mask = len(self.beats) * [True]
        beats_to_aggregate = self.beats[beats_to_aggregate_mask]
        beats_data = np.array([beat.data for beat in beats_to_aggregate])
        beats_times = np.array([beat.time for beat in beats_to_aggregate])
        agg_beat_data, agg_beat_time = beats_data.mean(axis=0), beats_times.mean(axis=0)
        agg_fs = len(agg_beat_data) / (agg_beat_time[-1] - agg_beat_time[0])
        BeatClass = beats_to_aggregate[0].__class__
        agg_beat = BeatClass("agg_beat", agg_beat_data, agg_fs, start_sec=0)
        self.agg_beat = agg_beat
        return self.agg_beat

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
    def __init__(self, name, data, fs, start_sec, beat_num=0):
        super().__init__(name, data, fs, start_sec, beat_num)

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
            self.plot_crit_points(ax=ax)
        return feats

    def plot_crit_points(self, ax=None):
        xrange = self.duration
        yrange = self.data.max() - self.data.min()

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 7))

        self.plot(ax=ax)
        for name, loc in self.crit_points.items():
            t, val = self.time[loc], self.data[loc]
            ax.scatter(t, val, label=name, s=160)
            ax.annotate(name.capitalize(), (t + 0.012 * xrange, val + 0.012 * yrange), fontweight="bold", fontsize=18)
        plt.legend()
