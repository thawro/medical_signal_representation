from collections import OrderedDict

from msr.signals.base import MultiChannelSignal, Signal
from msr.signals.utils import parse_feats_to_array, parse_nested_feats


class EOGSignal(Signal):
    def __init__(self, name, data, fs, start_sec=0):
        super().__init__(name, data, fs, start_sec)

    # TODO: Add features

    def extract_features(self, return_arr=True, plot=False, parse=True):
        features = OrderedDict({"whole_signal_features": {}})
        features["whole_signal_features"] = super().extract_features(return_arr=False)
        features = parse_nested_feats(features) if parse else features
        features = {
            f"{self.name}__{feat_name}" if self.name not in feat_name else feat_name: feat_val
            for feat_name, feat_val in features.items()
        }
        self.feature_names = list(features.keys())
        # TODO: Add another features

        if return_arr:
            return parse_feats_to_array(features)
        return features


class MultiChannelEOGSignal(MultiChannelSignal):
    def __init__(self, signals):
        super().__init__(signals)


def create_multichannel_eog(data, fs):
    signals = OrderedDict({i + 1: EOGSignal(f"EOG_{i}", channel_data, fs) for i, channel_data in enumerate(data)})
    return MultiChannelEOGSignal(signals)
