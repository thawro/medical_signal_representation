from collections import OrderedDict

from scipy.signal import welch

from msr.signals.base import MultiChannelSignal, Signal
from msr.signals.utils import parse_feats_to_array, parse_nested_feats

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

    # TODO: Add another features

    def extract_features(self, return_arr=True, plot=False, parse=True):
        features = OrderedDict({"whole_signal_features": {}})
        features["whole_signal_features"] = super().extract_features(return_arr=False)
        features["whole_signal_features"]["rel_power_features"] = self.extract_relative_power_frequency_features()
        features = parse_nested_feats(features) if parse else features
        features = {f"{self.name}__{feat_name}": feat_val for feat_name, feat_val in features.items()}
        self.feature_names = list(features.keys())
        # TODO: Add another features

        if return_arr:
            return parse_feats_to_array(features)
        return features


class MultiChannelEEGSignal(MultiChannelSignal):
    def __init__(self, signals):
        super().__init__(signals)


def create_multichannel_eeg(data, fs):
    signals = OrderedDict({i + 1: EEGSignal(f"EEG_{i+1}", channel_data, fs) for i, channel_data in enumerate(data)})
    return MultiChannelEEGSignal(signals)
