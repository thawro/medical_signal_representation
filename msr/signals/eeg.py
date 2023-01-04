import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from msr.signals.base import MultiChannelSignal, Signal
from msr.signals.utils import BEAT_FIG_PARAMS, parse_feats_to_array

FREQ_BANDS = {"delta": [0.5, 4.5], "theta": [4.5, 8.5], "alpha": [8.5, 11.5], "sigma": [11.5, 15.5], "beta": [15.5, 30]}


class EEGSignal(Signal):
    def __init__(self, name, data, fs, start_sec=0):
        super().__init__(name, data, fs, start_sec)
        self.units = "Voltage [V]"
        self.feature_extraction_funcs.update(
            {
                "frequency": self.extract_frequency_features,
                # TODO
            }
        )

    def extract_frequency_features(self, return_arr=False, fmin=0.5, fmax=30, plot=False, ax=None, **kwargs):
        return_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=BEAT_FIG_PARAMS["fig_size"])
            return_fig = True
        freqs, pxx = self.psd(fmin, fmax, normalize=True, plot=plot, ax=ax)
        features = {
            f"{band}_power": np.mean(pxx[(freqs >= fmin) & (freqs < fmax)]) for band, (fmin, fmax) in FREQ_BANDS.items()
        }
        if return_arr:
            return parse_feats_to_array(features)
        if return_fig:
            return features, fig
        return features

    # TODO
    def extract_xyz_features(self, return_arr=False, **kwargs):
        features = {}
        # TODO
        if return_arr:
            return parse_feats_to_array(features)
        return features


class MultiChannelEEGSignal(MultiChannelSignal):
    def __init__(self, signals):
        super().__init__(signals)


def create_multichannel_eeg(data, fs):
    signals = {i + 1: EEGSignal(f"EEG_{i+1}", channel_data, fs) for i, channel_data in enumerate(data)}
    return MultiChannelEEGSignal(signals)
