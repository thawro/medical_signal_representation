from collections import OrderedDict

import numpy as np

from msr.signals.base import MultiChannelPeriodicSignal
from msr.signals.ecg import ECGSignal
from msr.signals.ppg import PPGSignal
from msr.signals.utils import parse_feats_to_array


class MimicMeasurement(MultiChannelPeriodicSignal):
    def __init__(self, ppg: np.ndarray, ecg: np.ndarray, fs: float = 125):
        signals = OrderedDict(
            {
                "ppg": PPGSignal("PLETH", ppg, fs),
                "ecg": ECGSignal("ECG II", ecg, fs),
            }
        )
        super().__init__(signals)
        self.feature_extraction_funcs.update({"ptt_features": self.extract_ptt_features})

    def extract_ptt_features(self, return_arr=True, plot=False):
        features = OrderedDict({"ptt": 0})
        if return_arr:
            return parse_feats_to_array(features)
        return features
