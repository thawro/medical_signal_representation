from collections import OrderedDict

import numpy as np

from msr.signals.base import MultiChannelSignal
from msr.signals.eeg import EEGSignal
from msr.signals.eog import EOGSignal
from msr.signals.utils import parse_feats_to_array


class SleepEDFMeasurement(MultiChannelSignal):
    def __init__(self, eeg_0: np.ndarray, eeg_1: np.ndarray, eog: np.ndarray, fs: float = 100):
        signals = OrderedDict(
            {
                "eeg_0": EEGSignal("EEG Fpz-Cz", eeg_0, fs),
                "eeg_1": EEGSignal("EEG Pz-Oz", eeg_1, fs),
                "eog": EOGSignal("EOG horizontal", eog, fs),
            }
        )
        super().__init__(signals)
        self.feature_extraction_funcs.update({"some_features": self.extract_some_features})  # TODO

    def extract_some_features(self, return_arr=True, plot=False):
        # TODO
        features = OrderedDict({"some_feature": 0})
        if return_arr:
            return parse_feats_to_array(features)
        return features
