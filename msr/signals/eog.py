import neurokit2 as nk
import numpy as np

from msr.signals.base import (
    BeatSignal,
    MultiChannelSignal,
    PeriodicSignal,
    find_intervals_using_most_dominant_freq,
)
from msr.signals.utils import parse_feats_to_array, parse_nested_feats


class EOGSignal(PeriodicSignal):
    def __init__(self, name, data, fs, start_sec=0):
        super().__init__(name, data, fs, start_sec)
        self.units = "Voltage [V]"
        self.feature_extraction_funcs.update({"eye_blink": self.extract_eyeblink_features})

    @property
    def BeatClass(self):
        return EOGBeat

    @property
    def cleaned(self):
        return nk.eog_clean(self.data, sampling_rate=self.fs)

    def find_peaks(self):
        peaks = nk.eog_peaks(self.cleaned, sampling_rate=self.fs, method="mne")[1]["EOG_Blinks"]
        if len(peaks) == 0:
            peaks = super().find_peaks()  # with scipy.signal.find_peaks
        return peaks

    def _get_beats_intervals(self):
        try:
            eog_signals, peaks = nk.eog_process(self.data, sampling_rate=self.fs)
            events = nk.epochs_create(
                eog_signals["EOG_Clean"],
                peaks["EOG_Blinks"],
                sampling_rate=self.fs,
                epochs_start=-0.3,
                epochs_end=0.7,
            )
            intervals = []
            for idx, event in events.items():
                intervals.append((event["Index"].iloc[0], event["Index"].iloc[-1]))
            intervals = np.array(intervals)
            intervals = intervals[intervals[:, 0] > 0]
        except Exception:
            intervals = find_intervals_using_most_dominant_freq(self)
        return intervals

    def extract_eyeblink_features(self, return_arr=True, plot=False, **kwargs):
        # info = nk.eog_features(eog_cleaned, peaks, sampling_rate=100)

        peaks = self.peaks
        if len(peaks) == 0:
            freq, ibi_mean, ibi_std, mean_val = np.nan, np.nan, np.nan, np.nan
        else:
            vals = self.data[peaks]
            times = self.time[peaks]
            ibi = np.diff(times)
            ibi_mean = np.mean(ibi)
            ibi_std = np.std(ibi)
            mean_val = np.mean(vals)
            freq = self.duration / len(peaks)
        features = {"freq": freq, "ibi_mean": ibi_mean, "ibi_std": ibi_std, "blink_val": mean_val}
        if return_arr:
            return parse_feats_to_array(features)
        return features


class EOGBeat(BeatSignal):
    def __init__(self, name, data, fs, start_sec, beat_num=0, is_valid=True):
        super().__init__(name, data, fs, start_sec, beat_num, is_valid)


class MultiChannelEOGSignal(MultiChannelSignal):
    def __init__(self, signals):
        super().__init__(signals)
