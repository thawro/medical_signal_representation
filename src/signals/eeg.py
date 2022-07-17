from collections import OrderedDict

from scipy.signal import welch

from signals.base import MultiChannelSignal, Signal
from signals.utils import (
    calculate_area,
    calculate_energy,
    calculate_slope,
    create_new_obj,
    lazy_property,
    parse_feats_to_array,
    parse_nested_feats,
)

FREQ_BANDS = {"delta": [0.5, 4.5], "theta": [4.5, 8.5], "alpha": [8.5, 11.5], "sigma": [11.5, 15.5], "beta": [15.5, 30]}


class EEGSignal(Signal):
    def __init__(self, name, data, fs, start_sec=0):
        super().__init__(name, data, fs, start_sec)

    def extract_relative_power_frequency_features(self, fmin=0.5, fmax=30):
        freqs, pxx = welch(self.data, fs=self.fs)
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        freqs, pxx = freqs[freq_mask], pxx[freq_mask]
        pxx /= pxx.sum()
        return {band: pxx[(freqs >= fmin) & (freqs < fmax)].mean() for band, (fmin, fmax) in FREQ_BANDS.items()}

    def extract_features(self, return_arr=True, flatten=True, plot=False):
        features = OrderedDict({"whole_signal_features": {}})
        features["whole_signal_features"] = super().extract_features(return_arr=False)
        features["whole_signal_features"]["rel_power_features"] = self.extract_relative_power_frequency_features()
        if return_arr:
            return parse_feats_to_array(features)
        if flatten:
            return parse_nested_feats(features)
        return features


class MultiChannelEEGSignal(MultiChannelSignal):
    def __init__(self, signals):
        super().__init__(signals)
