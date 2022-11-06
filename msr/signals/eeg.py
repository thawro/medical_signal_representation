from collections import OrderedDict

from scipy.signal import welch

from msr.signals.base import MultiChannelSignal, Signal
from msr.signals.utils import parse_feats_to_array, parse_nested_feats

FREQ_BANDS = {"delta": [0.5, 4.5], "theta": [4.5, 8.5], "alpha": [8.5, 11.5], "sigma": [11.5, 15.5], "beta": [15.5, 30]}


class EEGSignal(Signal):
    def __init__(self, name, data, fs, start_sec=0):
        super().__init__(name, data, fs, start_sec)
        self.feature_extraction_funcs.update(
            {
                "relative_power_frequency_features": self.extract_relative_power_frequency_features,
            }
        )

    def extract_relative_power_frequency_features(self, return_arr=False, fmin=0.5, fmax=30, **kwargs):
        freqs, pxx = welch(self.data, fs=self.fs)
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        freqs, pxx = freqs[freq_mask], pxx[freq_mask]
        pxx /= pxx.sum()
        features = OrderedDict(
            {band: pxx[(freqs >= fmin) & (freqs < fmax)].mean() for band, (fmin, fmax) in FREQ_BANDS.items()}
        )
        if return_arr:
            return parse_feats_to_array(features)
        return features

    # TODO: Add another features


class MultiChannelEEGSignal(MultiChannelSignal):
    def __init__(self, signals):
        super().__init__(signals)


def create_multichannel_eeg(data, fs):
    signals = OrderedDict({i + 1: EEGSignal(f"EEG_{i+1}", channel_data, fs) for i, channel_data in enumerate(data)})
    return MultiChannelEEGSignal(signals)
