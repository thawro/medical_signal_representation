from collections import OrderedDict

from msr.signals.base import MultiChannelSignal, Signal
from msr.signals.utils import parse_feats_to_array, parse_nested_feats


class EOGSignal(Signal):
    def __init__(self, name, data, fs, start_sec=0):
        super().__init__(name, data, fs, start_sec)
        self.feature_extraction_funcs.update(
            {
                # TODO
            }
        )

    # TODO
    def extract_xyz_features(self, return_arr=False, **kwargs):
        features = OrderedDict({})
        # TODO
        if return_arr:
            return parse_feats_to_array(features)
        return features


class MultiChannelEOGSignal(MultiChannelSignal):
    def __init__(self, signals):
        super().__init__(signals)
