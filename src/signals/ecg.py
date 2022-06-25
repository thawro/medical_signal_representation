import matplotlib.pyplot as plt
import numpy as np
from biosppy.signals.ecg import ecg
from scipy.signal import find_peaks

from signals.base import BeatSignal, PeriodicSignal


class ECGSignal(PeriodicSignal):
    def __init__(self, name, data, fs, start_sec=0):
        super().__init__(name, data, fs, start_sec)
        self.ecg_info = ecg(data, fs, show=False)
        self.rpeaks = None

    def find_rpeaks(self):
        self.rpeaks = self.ecg_info["rpeaks"]
        return self.rpeaks

    def get_beats(self, resample_beats=True, n_samples=100, validate=True, plot=False):
        if self.rpeaks is None:
            self.find_rpeaks()
        beats = self.ecg_info["templates"]
        beats_time = self.ecg_info["templates_ts"]
        beats_fs = len(beats_time) / beats_time[-1]
        self.beats = []
        for i, beat in enumerate(beats):
            self.beats.append(ECGBeat(f"beat_{i}", beat, beats_fs, start_sec=i * beats_fs))
        self.beats = np.array(self.beats)
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

    def find_r_peak(self):
        self.r_loc = self.data.argmax()
        self.r_val = self.data[self.r_loc]
        self.r_time = self.time[self.r_loc]
        return self.r_loc

    def find_p_peak(self):
        zero_to_r_data = self.data[: self.r_loc]
        peaks, _ = find_peaks(zero_to_r_data)
        biggest_peak_idx = np.array([zero_to_r_data[peak] for peak in peaks]).argmax()

        self.p_loc = peaks[biggest_peak_idx]  # zero_to_r_data.argmax()
        self.p_val = self.data[self.p_loc]
        self.p_time = self.time[self.p_loc]
        return self.p_loc

    def find_q_notch(self):
        p_to_r_data = self.data[self.p_loc : self.r_loc]
        self.q_loc = p_to_r_data.argmin() + self.p_loc
        self.q_val = self.data[self.q_loc]
        self.q_time = self.time[self.q_loc]
        return self.q_loc

    def find_s_notch(self):
        r_to_end_data = self.data[self.r_loc :]
        troughs, _ = find_peaks(-r_to_end_data)
        self.s_loc = troughs[0] + self.r_loc
        self.s_val = self.data[self.s_loc]
        self.s_time = self.time[self.s_loc]
        return self.s_loc

    def find_t_peak(self):
        s_to_end_data = self.data[self.s_loc :]
        peaks, _ = find_peaks(s_to_end_data)
        biggest_peak_idx = np.array([s_to_end_data[peak] for peak in peaks]).argmax()

        self.t_loc = peaks[biggest_peak_idx] + self.s_loc  # zero_to_r_data.argmax()
        self.t_val = self.data[self.t_loc]
        self.t_time = self.time[self.t_loc]
        return self.t_loc

    def extract_agg_beat_features(self, plot=True, ax=None):
        self.find_r_peak()
        self.find_p_peak()
        self.find_q_notch()
        self.find_s_notch()
        self.find_t_peak()

        crit_points = {"p": self.p_loc, "q": self.q_loc, "r": self.r_loc, "s": self.s_loc, "t": self.t_loc}
        if plot and ax is None:
            fig, ax = plt.subplots(figsize=(12, 7))
            ax.plot(self.time, self.data)

        feats = {}
        for name, loc in crit_points.items():
            t, val = self.time[loc], self.data[loc]
            feats[f"{name}_loc"] = loc
            feats[f"{name}_time"] = t
            feats[f"{name}_val"] = val
            if plot:
                ax.scatter(t, val, label=name, s=120)
                ax.legend()
        return feats
